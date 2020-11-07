import numpy as np
import torch
import argparse
import yaml
import wandb
import transformers
from module import CQAModel
from transformers import BertTokenizer
from train_utils import train
from extract_feature import *
from utils import *

def main():
    # model: paragraph, question_answering module

    config = yaml.safe_load(open("train_config.yaml","r"))
    
    set_seed(config['seed'])
    model= CQAModel(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    wandb.init(project="doqa", name=config['exp_name'])
    wandb.config.update(config)

    wandb.watch(model)

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, config['train_feature_file'], config['eval_feature_file'], config, tokenizer, wandb)

if __name__ == '__main__':
    main()
