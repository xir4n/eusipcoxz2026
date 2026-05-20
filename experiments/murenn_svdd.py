import os
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchmetrics import AUROC
import argparse
import pandas as pd

from .experiment_base import BaseExperiment
import models
from data import INVERSE_SNR_MAP, INVERSE_CLASS_MAP

class MurennExperiment(BaseExperiment, pl.LightningModule):
    def __init__(self, configuration_dict):
        super().__init__(configuration_dict)
        self.save_hyperparameters()
        self.network = getattr(models, self.configuration_dict['arch'])(**self.configuration_dict['model_settings'])
        self.val_outputs = {'score': [], 'label': []}
        self.auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.register_buffer('svdd_c', None)

    def svdd_c_init(self):
        print('Initializing SVDD center c...')
        self.dataset.setup()
        device = self.device
        dataloadr = self.dataset.train_dataloader()
        with torch.no_grad():
            scores = []
            for i, batch in enumerate(dataloadr):
                x = batch["sample"].to(device)
                outputs = self.network(x)
                scores.append(outputs)
            scores = torch.cat(scores, dim=0)
        c = torch.mean(scores, dim=0, keepdim=True)
        self.svdd_c = c

    def on_load_checkpoint(self, checkpoint):
        self.svdd_c = checkpoint['svdd_c']

    def on_save_checkpoint(self, checkpoint):
        checkpoint['svdd_c'] = self.svdd_c

    def forward(self, batch):
        pred = self.network(batch['sample'])
        batch['scores'] = pred
        return batch
    
    def svdd_loss(self, x):
        """
        Calculate the SVDD loss.
        """
        if self.svdd_c is None:
            raise ValueError("SVDD hypersphere center not set.")
        
        dist = torch.sum((x - self.svdd_c) ** 2, dim=1)
        loss = torch.mean(dist)
        return loss

    def on_fit_start(self):
        if self.svdd_c is None:
            self.svdd_c_init()

    def training_step(self, batch):
        batch = self(batch)
        loss = self.svdd_loss(batch['scores'])
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
       batch = self(batch)
       dist = torch.sum((batch['scores'] - self.svdd_c) ** 2, dim=1)
       self.val_outputs['score'].append(dist.view(-1, 1).detach())
       self.val_outputs['label'].append(batch['label'].view(-1, 1).detach())

    def on_validation_epoch_end(self):
       scores = torch.cat(self.val_outputs['score'], dim=0)
       labels = torch.cat(self.val_outputs['label'], dim=0)
       auroc = self.auroc(scores, labels)
       self.auroc.reset()
       self.log('val_auroc', auroc)
       self.val_outputs['score'].clear()
       self.val_outputs['label'].clear()

    def on_test_start(self):
        if self.svdd_c is None:
            self.svdd_c_init()
        self.file_names = []
        self.scores = []
        self.label = []
        pass

    def test_step(self, batch, batch_idx):
        batch = self(batch)
        dist = torch.sum((batch['scores'] - self.svdd_c) ** 2, dim=1)
        self.file_names.extend(batch['file_id'])
        self.scores.extend(dist.detach().cpu().numpy())
        self.label.extend(batch['label'].detach().cpu().numpy())



    def on_test_end(self):
        auroc = self.test_auroc(torch.Tensor(self.scores), torch.Tensor(self.label))
        print(f"auroc: {auroc}")
        self.logger.experiment.add_scalar("test_auroc", auroc)
        self.test_auroc.reset()
        df = pd.DataFrame({
            'file': self.file_names,
            'score': self.scores,
            'label': self.label,
        })
        csv_path = os.path.join(
            self.configuration_dict['model_save_path'],  
            'evaluate.csv'
)
        df.to_csv(csv_path)

        print('Test results saved to:', csv_path)
        self.flag = 0
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), **self.configuration_dict['optimizer']['kwargs'])
        self.warmup_epochs = 5
        # Warmup scheduler
        def warmup_scheduler(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler)

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs)

        # Combine schedulers
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs]
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]


def configuration(snr, machine_type, machine_id, split_idx, lr, batch_size, num_samples, save_folder, J, Q, T, J_phi, mode, use_power):

    return {
        "arch": "LearnableScattering",
        "model_save_path" : os.path.join(save_folder),
        "dataset_settings" : {
            "class_name": "MIMII_OneClass",
            "kwargs": {
                "snr": snr,
                "machine_type": machine_type,
                "machine_id": machine_id,
                "normalize_raw": True,
                "num_samples": num_samples,
                "data_path" : "/LAB-DATA/GLiCID/users/cz7tygkr@cnrs.fr/MIMII",
                "batch_size": batch_size,
                "num_workers": 1,
                "split_idx": split_idx,
            }
        },

        "model_settings" : {
            "J": J,
            "Q": Q,
            "T": T,
            "J_phi": J_phi,
            "use_power": use_power,
            "mode" : mode,
        },


        "lr_scheduler" : {
            'class': 'torch.optim.lr_scheduler.ExponentialLR',
            'args': [
                '@optimizer',
            ],
            'kwargs': {
                'gamma': 0.98,
            }
        },

        "optimizer" : {
            'class': 'torch.optim.Adam',
            'kwargs': {
                'lr': lr,
                'betas': (0.9, 0.999),
                'amsgrad': False,
                'weight_decay': 0,
            }
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=int, default=0, choices=[0, 6, -6], help='Signal-to-Noise Ratio')
    parser.add_argument('--machine_type', type=int, default=0, choices=[0, 1, 2, 3], help='Machine type')
    parser.add_argument('--machine_id', type=int, default=0, choices=[0, 2, 4, 6], help='Machine ID')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use for training')
    parser.add_argument('--save_folder', type=str, default='test')
    parser.add_argument('--J', type=int, nargs='*', default=[8, 6])
    parser.add_argument('--Q', type=int, nargs='*', default=[8, 1])
    parser.add_argument('--T', type=int, nargs='*', default=[32, 1])
    parser.add_argument('--J_phi', type=int, nargs='*', default=[4, 10])
    parser.add_argument('--mode', type=str, default='avg', choices=['avg', 'grouped', 'grouped_relu', 'residual', 'fc', 'fc_relu'])
    parser.add_argument('--use_power', action='store_true')
    parser.add_argument('--split_idx', type=int, default=None)
    config = configuration(
        snr=parser.parse_args().snr,
        machine_type=parser.parse_args().machine_type,
        machine_id=parser.parse_args().machine_id,
        split_idx=parser.parse_args().split_idx,
        lr=parser.parse_args().lr,
        batch_size=parser.parse_args().batch_size,
        num_samples=parser.parse_args().num_samples,
        save_folder=parser.parse_args().save_folder,
        J=parser.parse_args().J,
        Q=parser.parse_args().Q,
        T=parser.parse_args().T,
        J_phi=parser.parse_args().J_phi,
        mode=parser.parse_args().mode,
        use_power=parser.parse_args().use_power,
    )

    experiment = MurennExperiment(config)
    experiment.run()
