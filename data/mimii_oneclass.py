import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import random

from .base_data_set import BaseDataSet

class MIMII_OneClass(pl.LightningDataModule):
    def __init__(
            self,
            snr=0,
            machine_type=0,
            machine_id=0,
            split_idx=0,
            normalize_raw=True,
            num_samples=None,
            data_path=None,
            batch_size=32,
            num_workers=1,
    ):
        super(MIMII_OneClass, self).__init__()

        self.snr = snr
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_samples = num_samples

        self.base_set =  BaseDataSet(
            snr=self.snr,
            machine_type=self.machine_type,
            machine_id=self.machine_id,
            normal=True,
            label=0,
            data_path=self.data_path,
            normalize_raw=normalize_raw,
        )
        self.test_negative = BaseDataSet(
            snr=self.snr,
            machine_type=self.machine_type,
            machine_id=self.machine_id,
            normal=False,
            label=1,
            data_path=self.data_path,
            normalize_raw=normalize_raw,
        )
    
    def setup(self, stage=None):
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(42)

        num_folds = 5
        assert 0 <= self.split_idx < num_folds, "split_idx must be in [0, 4]"

        dataset_size = len(self.base_set)

        indices = torch.randperm(dataset_size, generator=generator).tolist()

        fold_size = dataset_size // num_folds
        fold_sizes = [fold_size] * num_folds

        for i in range(dataset_size % num_folds):
            fold_sizes[i] += 1

        start = sum(fold_sizes[:self.split_idx])
        end = start + fold_sizes[self.split_idx]

        test_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        self.train_positive = torch.utils.data.Subset(self.base_set, train_indices)
        self.test_positive = torch.utils.data.Subset(self.base_set, test_indices)

        if self.n_samples is not None:
            assert self.n_samples > 0
            num_samples = min(self.n_samples, len(self.train_positive))

            sample_indices = torch.randperm(
                len(self.train_positive),
                generator=generator
            )[:num_samples]

            self.train_positive = torch.utils.data.Subset(
                self.train_positive,
                sample_indices
            )

        self.test_set = torch.utils.data.ConcatDataset([
            self.test_positive,
            self.test_negative
        ])

    
    def train_dataloader(self):
        return DataLoader(
            self.train_positive, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    
    def val_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

