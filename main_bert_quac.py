import numpy as np
import torch
import argparse
import yaml
import wandb
import transformers

from module import BERTQA_memory, BERTQA, BERTQA_original, BERTQA_memoryHistory, BERTQA_memory_HAE, BERTQA_HHF
from transformers import BertTokenizer, BertConfig
from utils.train_utils_quac import train
from extract_feature.extract_feature_bert_quac_truncated import *
from utils.utils import *


# os.environ['WANDB_MODE'] = 'dryrun'


def main():
    # model: paragraph, question_answering module

    parser = argparse.ArgumentParser(description='Argument Parser for HistoryQA project.')
    parser.add_argument("--config")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config,"r"))
    
    set_seed(config['seed'])
    model= eval(config['model'])(config)
    tokenizer = eval(config["pretrained_tokenizer"]).from_pretrained(config["pretrained_name"])

    wandb.init(project="quac_official_twcc", name=config['exp_name'],settings=wandb.Settings(start_method="fork"))
    wandb.config.update(config)

    wandb.watch(model)

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, config['train_feature_file'], config['eval_feature_file'], config['eval_json'], config, tokenizer, wandb)

if __name__ == '__main__':
    main()
