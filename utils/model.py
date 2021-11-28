from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

pl.seed_everything(100)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride, dilation: int, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (kernel_size, kernel_size), (stride, stride), padding, (dilation, dilation)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# Not technically a residual block because it uses concatenation instead of identity mapping, but the authors call it
# anyways
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
    def __init__(self, hsi_bands: int, lidar_bands: int, num_classes: int):
        super().__init__()
        self.hsi_bands = hsi_bands
        self.lidar_bands = lidar_bands
        self.num_classes = num_classes
        self.hsi_feature = HsExtractor(in_channels=hsi_bands)
        self.hsi_attn = SpectralAttention(in_channels=hsi_bands)
        self.lidar_attn = SpatialAttention(in_channels=lidar_bands)
        self.modality_attn = ModalityAttention(in_channels=(hsi_bands + lidar_bands + 1024 + 1024))
        self.classifier = Classifier(in_channels=1024, num_classes=num_classes)

    def forward(self, x):
        x_hsi, x_lidar = x
        feat_hsi = self.hsi_feature(x_hsi)
        mask_spectral = self.hsi_attn(x_hsi)
        mask_spatial = self.lidar_attn(x_lidar)

        feat_spatial = feat_hsi * mask_spatial
        feat_spectral = feat_hsi * mask_spectral
        feat_fused = torch.cat([x_hsi, x_lidar, feat_spectral, feat_spatial], dim=1)
        feat_fused = self.modality_attn(feat_fused)
        classification = self.classifier(feat_fused)

        return classification


if __name__ == "__main__":
    model = FusAtNet(hsi_bands=144, lidar_bands=1, num_classes=15)
    test = (torch.randn((16, 144, 11, 11)), torch.randn(16, 1, 11, 11))
    model(test)
