import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


HOUSTON_CLASSES = (
    "Healthy Grass", "Stressed Grass", "Synthetic Grass", "Trees", "Soil", "Water", "Residential",
    "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
)


class HoustonDataset(LightningDataModule):
    def __init__(self):
        super().__init__()


class HoustonPatches(LightningDataModule):
    def __init__(self, train_patches, train_labels, test_patches, test_labels, batch_size, class_names=HOUSTON_CLASSES):
        super().__init__()
        self.train_patches = torch.from_numpy(np.load(train_patches).transpose((0, 3, 1, 2)).astype(np.float32))
        self.test_patches = torch.from_numpy(np.load(test_patches).transpose((0, 3, 1, 2)).astype(np.float32))
        self.train_labels = torch.from_numpy(np.load(train_labels).astype(int))
        self.test_labels = torch.from_numpy(np.load(test_labels).astype(int))
        self.train_labels -= self.train_labels.min()
        self.test_labels -= self.test_labels.min()

        self.class_names = class_names
        self.batch_size = batch_size

        self.train_dataset = TensorDataset(self.train_patches, self.train_labels)
        self.test_dataset = TensorDataset(self.test_patches, self.test_labels)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)