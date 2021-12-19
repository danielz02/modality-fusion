import os
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
from typing import List, Tuple
import pytorch_lightning as pl
import torch.nn.functional as F
from dataset import HoustonPatches
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import precision_recall_fscore_support
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride, dilation: int, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), padding, (dilation, dilation)
            ),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: List[int], pooling: bool):
        super().__init__()
        self.pooling = pooling
        self.conv1 = ConvBlock(
            in_channels=n_channels[0], out_channels=n_channels[1], kernel_size=3, stride=1, dilation=1, padding="same"
        )
        self.conv2 = ConvBlock(
            in_channels=n_channels[1], out_channels=n_channels[2], kernel_size=3, stride=1, dilation=1, padding="same"
        )
        if self.pooling:
            self.mp = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.conv2(x)
        x = torch.cat([identity, x], dim=1)
        if self.pooling:
            x = self.mp(x)

        return x


class HsExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.hs_conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding="same")
        )

    def forward(self, x):
        return self.hs_conv(x)


class SpectralAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.res1 = ResidualBlock(n_channels=[in_channels, 256, 256], pooling=True)
        self.res2 = ResidualBlock(n_channels=[256 * 2, 256, 256], pooling=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels=256 * 2, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding="same")
        )
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv(x)
        x = self.mp(x)
        x = self.gap(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.res1 = ResidualBlock(n_channels=[in_channels, 128, 128], pooling=False)
        self.res2 = ResidualBlock(n_channels=[128 * 2, 128, 256], pooling=False)
        self.conv = nn.Sequential(
            ConvBlock(in_channels=256 + 128, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding="same")
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv(x)

        return x


class ModalityAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding="same"),
        )

        self.attn = nn.Sequential(
            ResidualBlock(n_channels=[in_channels, 128, 128], pooling=False),
            ResidualBlock(n_channels=[128 * 2, 128, 256], pooling=False),
            ConvBlock(in_channels=256 + 128, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="same"),
            ConvBlock(in_channels=256, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding="same"),
        )

    def forward(self, x):
        feat = self.conv(x)
        mask = self.attn(x)

        return feat * mask


class Classifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="valid"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="valid"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="valid"),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding="valid"),
            ConvBlock(in_channels=256, out_channels=1024, kernel_size=3, stride=1, dilation=1, padding="valid"),
        )
        self.clf = nn.Conv2d(
            in_channels=1024, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1), padding="valid"
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.clf(x).reshape(len(x), self.num_classes)

        return x


class FusAtNet(pl.LightningModule):
    def __init__(self, hsi_bands: int, lidar_bands: int, class_names: List[str], lr: float = 1e-3):
        super().__init__()
        self.learning_rate = lr
        self.hsi_bands = hsi_bands
        self.lidar_bands = lidar_bands
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.hsi_feature = HsExtractor(in_channels=hsi_bands)
        self.hsi_attn = SpectralAttention(in_channels=hsi_bands)
        self.lidar_attn = SpatialAttention(in_channels=lidar_bands)
        self.modality_attn = ModalityAttention(in_channels=(hsi_bands + lidar_bands + 1024 + 1024))
        self.classifier = Classifier(in_channels=1024, num_classes=len(class_names))

        self.__initialize_weights()
        self.save_hyperparameters()

    def forward(self, x):
        x_hsi, x_lidar = x[:, :-1, ...], x[:, -1, ...].unsqueeze(1)
        feat_hsi = self.hsi_feature(x_hsi)
        mask_spectral = self.hsi_attn(x_hsi)
        mask_spatial = self.lidar_attn(x_lidar)

        feat_spatial = feat_hsi * mask_spatial
        feat_spectral = feat_hsi * mask_spectral
        feat_fused = torch.cat([x_hsi, x_lidar, feat_spectral, feat_spatial], dim=1)
        feat_fused = self.modality_attn(feat_fused)
        classification = self.classifier(feat_fused)

        return classification

    def configure_optimizers(self):
        return torch.optim.AdamW(lr=self.learning_rate, params=self.parameters(), weight_decay=1e-2)

    def __initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __common_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y_true = batch
        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y_true)

        return loss, y_pred, y_true

    def __evaluation(self, y_pred, y_true, stage):
        assert stage == "train" or stage == "validation" or stage == "test"

        y_prob = F.softmax(y_pred, dim=1).detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy().argmax(axis=1)
        y_true = y_true.detach().cpu().numpy()

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        balanced_accuracy = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
        clf_report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_pred=y_pred, y_true=y_true, average="weighted"
        )

        self.log(f"{stage}/fscore", f_score)
        # self.log(f"{stage}_report", clf_report)
        self.log(f"{stage}/precision", precision)
        self.log(f"{stage}/recall", recall)
        self.log(f"{stage}/accuracy", accuracy)
        self.log(f"{stage}/accuracy_balanced", balanced_accuracy)
        self.logger.experiment.log({
            f"{stage}/cm": wandb.plot.confusion_matrix(probs=y_prob, y_true=y_true, class_names=self.class_names),
            "global_step": self.global_step
        })

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        loss, y_pred, y_true = self.__common_step(train_batch)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "y_pred": y_pred,
            "y_true": y_true
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_pred = torch.cat([x["y_pred"] for x in outputs]).reshape(-1, self.num_classes)
        y_true = torch.cat([x["y_true"] for x in outputs]).reshape(-1)

        self.__evaluation(y_pred, y_true, "train")

    def validation_step(self, test_batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        _, y_pred, y_true = self.__common_step(test_batch)

        return y_pred, y_true

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        y_true = torch.cat([x[1] for x in outputs]).reshape(-1)
        y_pred = torch.cat([x[0] for x in outputs]).reshape(-1, self.num_classes)
        loss = F.cross_entropy(y_pred, y_true).item()

        self.__evaluation(y_pred, y_true, "validation")
        self.log("validation/loss", loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="../ckpt")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint(
        monitor="train/accuracy", mode="max", dirpath=args.save_dir, auto_insert_metric_name=True,
        filename=f"{{epoch:02d}}-{{train/accuracy_balanced}}", verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor="train/accuracy", min_delta=0.005, patience=20, verbose=True, mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    logger = WandbLogger(project="fusion")
    trainer = Trainer(
        gpus=1, logger=logger, max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor], resume_from_checkpoint=args.resume_ckpt
    )

    data_module = HoustonPatches(
        train_patches="../data/train_patches.npy",
        train_labels="../data/train_labels.npy",
        test_patches="../data/test_patches.npy",
        test_labels="../data/test_labels.npy",
        batch_size=args.batch_size
    )

    model = FusAtNet(hsi_bands=144, lidar_bands=1, class_names=data_module.class_names, lr=0.000005)

    trainer.fit(model=model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module)
