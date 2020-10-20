import os
import torch
import numpy as np
import IPython
import pdb
import argparse
import tqdm
import apex
from apex import amp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from optimizer_utils import AdamW, get_linear_schedule_with_warmup
from train_utils import train
from tqdm import trange
from preprocess_rc import *
import yaml
from transformers import BertForQuestionAnswering

def main():
    train_args = yaml.safe_load(open("train_config.yaml","r"))

    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    wandb.init(project="doqa", name=train_args["exp_name"])
    wandb.config.update(train_args)

    train_args['device'] = torch.device("cuda")
    train_args['n_gpu'] = torch.cuda.device_count()

    train(model, "doqa_training_file", "doqa_validation_file", train_args, tokenizer, wandb)

if __name__ == "__main__":
    main()




