import numpy as np
import torch
import os
import argparse
import yaml
import wandb

import transformers
from module import BERTQA_memory, BERTQA, \
    BERTQA_memory_HAE, BERTQA_memoryHistory, BERTQA_HHF, BERTQA_original
from transformers import BertTokenizer, BertConfig
from utils.evaluate_utils_quac_different_decode import evaluate 
from extract_feature.extract_feature_bert_quac_truncated_different_decode import *
from utils.utils import *

def main():

    parser = argparse.ArgumentParser(description='Argument Parser for HistoryQA project.')
    parser.add_argument("--config")
    args = parser.parse_args()
    eval_config = yaml.safe_load(open(args.config,"r"))
    
    if "bottleneck_size" not in list(eval_config.keys()):
        eval_config['bottleneck_size'] = 32

    saved_directory = eval_config['saved_directory']
    sub_directory_list = [x[0] for x in os.walk(saved_directory)]
    sub_directory_list = sub_directory_list[1:]
    numbers = [int(x.split("checkpoint-")[-1]) for x in sub_directory_list]
    checkpoint_step_str = str(max(numbers))
    eval_config['checkpoint_step'] = checkpoint_step_str
    checkpoint_step = "checkpoint-{}".format(eval_config['checkpoint_step'])
    dir_path = os.path.join(saved_directory, checkpoint_step)
    config = torch.load(os.path.join(dir_path,"training_args.bin"))

    set_seed(config['seed'])
    model = eval(config['model'])(config)
    model.load_state_dict(torch.load(os.path.join(dir_path, "model.pt"), map_location="cpu"),strict=False)

    tokenizer = BertTokenizer.from_pretrained(dir_path)

    wandb.init(project="quac_different_eval_official", name=eval_config['exp_name'])
    wandb.config.update(config)

    config['device'] = torch.device("cuda")
    config['n_gpu']  = torch.cuda.device_count()

    model = model.to(config['device'])


    results = evaluate(config, eval_config['eval_file'], eval_config['eval_json'], model, tokenizer,eval_config['beam_search'])

    record = {}
    for key, value in results.items():
        record["eval_{}".format(key)] = value

    print(record)

    wandb.log(record)

if __name__ == "__main__":
    main()