import torch
import numpy as np
import pytorch_lightning as pl
from pl.core.datamodule import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(self, args, config, **kwargs):
        super().__init__()
    
    def prepare_data(self):
        return
    def setup(self):
        return 
    def train_dataloader(self):
        return 
    def val_dataloader(self):
        return