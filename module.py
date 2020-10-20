import torch
import numpy as np
import torch.nn as nn



class Paragraph_Encoder(nn.Module):
    def __init__(self, hid_dim, n_layers):
        