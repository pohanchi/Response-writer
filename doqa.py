import numpy as np
import torch
import IPython, pdb
import functools
import tqdm
import wandb
import json
import copy
from datasets import load_dataset, list_datasets, load_metric
from transformers import BertModel, BertConfig, BertTokenizerFast, GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from torch import nn
from itertools import chain
import logging
import random
import re
import time
import torch_optimizer as optim
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased")
    squad_dataset = load_dataset("doqa", cache_dir="./doqa")

