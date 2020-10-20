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
from eval_utils import evaluate


def main():

    train_args = yaml.safe_load(open("eval_config.yaml","r"))

    wandb.init(project="doqa", name=train_args["exp_name"])
    train_args['device'] = torch.device("cuda")
    train_args['n_gpu'] = torch.cuda.device_count()

    model = BertForQuestionAnswering.from_pretrained("doqa_save_model_validation/checkpoint-5000/")
    model = model.to(train_args["device"])
    tokenizer = BertTokenizer.from_pretrained("doqa_save_model_validation/checkpoint-5000/")
    results = evaluate(train_args, "doqa_test_file_travel", model, tokenizer)
    record = {}
    for key, value in results.items():
        record["eval_{}".format(key)] = value
    wandb.log(record,step=0)

if __name__ == "__main__":
    main()