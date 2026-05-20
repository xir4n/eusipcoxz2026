import importlib
from abc import ABC, abstractmethod
import torch
import copy
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pandas as pd

import data

class Checkpoint(ModelCheckpoint):
    # This class saves the model's initial weights.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def on_train_start(self, trainer, pl_module):
        save_path = os.path.join(self.dirpath, "initial.ckpt")
        trainer.save_checkpoint(save_path)

class BaseExperiment(ABC, torch.nn.Module):

    def __init__(self, configuration_dict):
        super(BaseExperiment, self).__init__()
        self.configuration_dict = copy.deepcopy(configuration_dict)

        # Create model directory
        model_sav_path = self.configuration_dict['model_save_path']
        os.makedirs(model_sav_path, exist_ok=True)

        # Initialize dataset
        dataset_class = getattr(data, self.configuration_dict['dataset_settings']['class_name'])
        self.dataset = dataset_class(**self.configuration_dict['dataset_settings']['kwargs'])

        # Setup checkpoints and Tensorboard logger
        checkpoint_cb = Checkpoint(
            dirpath=model_sav_path,
            monitor="train_loss",
            save_last=True,
            filename="best",
            save_weights_only=False,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=model_sav_path,
        )
        self.max_epochs = 100
        # Setup trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            callbacks=[checkpoint_cb],
            logger=tb_logger,
            log_every_n_steps=1,
        )

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def training_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def test_step(self, batch):
        raise NotImplementedError()

    @abstractmethod
    def on_test_end(self):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), **self.configuration_dict['optimizer']['kwargs'])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            **self.configuration_dict['lr_scheduler']['kwargs']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

    def run(self):
        self.trainer.fit(self, self.dataset)
        self.trainer.test(self, self.dataset)

