import numpy as np
import torch
import argparse
import yaml
import wandb
import transformers
from module import CQAModel, BERTQA, BERTQA2, ALBERTQA2
from transformers import BertTokenizer, AlbertTokenizer
from train_utils import train
from extract_feature import *
from utils import *

def main():
    # model: paragraph, question_answering module
    parser = argparse.ArgumentParser(description='Argument Parser for HistoryQA project.')
    parser.add_argument("--config")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config,"r"))
    
    set_seed(config['seed'])
    model= ALBERTQA2(config)

    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    wandb.init(project="doqa", name=config['exp_name'])
    wandb.config.update(config)

    wandb.watch(model)

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, config['train_feature_file'], config['eval_feature_file'], config, tokenizer, wandb)

if __name__ == '__main__':
    main()
