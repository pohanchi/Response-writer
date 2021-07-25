import torch 

import numpy as np
import pytorch_lightning as pl


class HistoryQASystem(pl.LightningModule):
    def __init__(self, args, model_config, training_config):
        super().__init__()
        

        