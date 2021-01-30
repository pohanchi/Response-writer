import numpy as np
import torch
import argparse
import yaml
import wandb
import os
import transformers
from module import RobertaQA_memory, RobertaQA
from transformers import RobertaTokenizer
from train_utils import train
from extract_feature_roberta_doqa import *
from utils import *


# os.environ['WANDB_MODE'] = 'dryrun'

def main():
    # model: paragraph, question_answering module
    parser = argparse.ArgumentParser(description='Argument Parser for HistoryQA project.')
    parser.add_argument("--config")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config,"r"))
    
    set_seed(config['seed'])
    model= eval(config['model'])(config)

    if "pretrained_name" not in list(config.keys()):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(config['pretrained_name'])

    wandb.init(project="doqa_battleship_official", name=config['exp_name'])
    wandb.config.update(config)

    wandb.watch(model)

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, config['train_feature_file'], config['eval_feature_file'], config, tokenizer, wandb)

if __name__ == '__main__':
    main()
