import numpy as np
import torch
import argparse
import yaml
import wandb
import transformers
from module import BERTQA_memory, BERTQA, BERTQA_memory_coqa, BERTQA_coqa, BERTQA_memoryHistory_coqa, BERTQA_memory_HAE_coqa, BERTQA_memory23_coqa, BERTQA_original_coqa
from transformers import BertTokenizer
from utils.train_utils_coqa import train
from extract_feature.extract_feature_bert_coqa_truncated import *
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

    wandb.init(project="coqa_official_twcc", name=config['exp_name'])
    wandb.config.update(config)

    wandb.watch(model)

    config['device'] = torch.device("cuda")
    config['n_gpu'] = torch.cuda.device_count()

    train(model, config['train_feature_file'], config['eval_feature_file'], config['eval_json'], config, tokenizer, wandb)

if __name__ == '__main__':
    main()
