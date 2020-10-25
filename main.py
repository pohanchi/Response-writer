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

def main():
    # model: paragraph, question_answering module
    config = yaml.safe_load(open("train_config.yaml","r"))
    model= CQAModel(config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    wandb.init(project="doqa", name=config['exp_name'])
    wandb.config.update(config)

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, "doqa_train_file_cooking", "doqa_test_file_cooking", config, tokenizer, wandb)



if __name__ == '__main__':
    main()