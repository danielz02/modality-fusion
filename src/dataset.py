import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


HOUSTON_CLASSES = (
    "Healthy Grass", "Stressed Grass", "Synthetic Grass", "Trees", "Soil", "Water", "Residential",
    "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
)

HOUSTON_CLASSES_18 = (
    "Healthy Grass", "Stressed Grass", "Artificial Turf", "Evergreen trees", "Deciduous trees", "Bare earth", "Water",
    "Residential Buildings", "Non-residential Buildings", "Roads", "Sidewalks", "Crosswalks", "Major Thoroughfares",
    "Highways", "Railways", "Paved Parking Lots", "Unpaved Parking Lots", "Cars", "Trains", "Stadium Seats"
)


class HoustonDataset(LightningDataModule):
    def __init__(self):
        super().__init__()


class HoustonPatches(LightningDataModule):
    def __init__(self, train_patches, train_labels, test_patches, test_labels, batch_size, year=2013):
        super().__init__()
        assert year == 2013 or year == 2018
        if year == 2013:
            self.train_patches = torch.from_numpy(np.load(train_patches).transpose((0, 3, 1, 2)).astype(np.float32))
            self.test_patches = torch.from_numpy(np.load(test_patches).transpose((0, 3, 1, 2)).astype(np.float32))
        else:
            self.train_patches = torch.from_numpy(np.load(train_patches).astype(np.float32))
            self.test_patches = torch.from_numpy(np.load(test_patches).astype(np.float32))
        self.train_labels = torch.from_numpy(np.load(train_labels).astype(int))
        self.test_labels = torch.from_numpy(np.load(test_labels).astype(int))
        self.train_labels -= self.train_labels.min()
        self.test_labels -= self.test_labels.min()

        self.class_names = HOUSTON_CLASSES if year == 2013 else HOUSTON_CLASSES_18
        self.batch_size = batch_size

        self.train_dataset = TensorDataset(self.train_patches, self.train_labels)
        self.test_dataset = TensorDataset(self.test_patches, self.test_labels)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)


class ChampaignDataset(LightningDataModule):
    def __init__(self, patches, labels, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.patches = torch.from_numpy(np.load(patches).astype(np.float32))
        self.labels = torch.from_numpy(np.load(labels))
        self.labels[self.labels == 1] = 0
        self.labels[self.labels == 5] = 1
        self.dataset = TensorDataset(self.patches, self.labels)

        self.train_size = int(len(self.dataset) * 0.8)
        self.test_size = len(self.dataset) - self.train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [self.train_size, self.test_size]
        )
        self.class_names = ("Corn", "Soybean")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
