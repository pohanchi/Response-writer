import numpy as np
import torch
import argparse
import yaml
import wandb
import transformers
from module import BERTQA_initial, BERTQA_memory, BERTQA, BERTQA_memoryreuse, BERTQA_memory3, BARTQA
from module import ALBERTQA_memory, ALBERTQA
from module import RobertaQA_memory, RobertaQA
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, BartTokenizer
from train_utils import train
from extract_feature import *
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

    if "pretrained_name" in list(config.keys()):
        tokenizer = BartTokenizer.from_pretrained(config['pretrained_name'])
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    wandb.init(project="doqa_battleship_official", name=config['exp_name'])
    wandb.config.update(config)

    wandb.watch(model,log="all")

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, config['train_feature_file'], config['eval_feature_file'], config, tokenizer, wandb)

if __name__ == '__main__':
    main()
