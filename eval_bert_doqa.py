import numpy as np
import torch
import os
import argparse
import yaml
import wandb
import IPython
import pdb
import transformers

from module import BERTQA_initial, BERTQA_memory, BERTQA, BERTQA_memoryreuse, \
    BERTQA_memory3, BERTQA_memory4, BERTQA_memory5, BERTQA_memory6, BERTQA_memory10, BERTQA_memory11, \
    BERTQA_memory12, BERTQA_memory115, BERTQA_memory13, BERTQA_memory14, BERTQA_memory15, BERTQA_memory16, \
    BERTQA_memory17, BERTQA_memory18, BERTQA_memory_HAE, BERTQA_memoryHistory, BERTQA_memory20, BERTQA_memory21, BERTQA_memory22, \
    BERTQA_memory23, BERTQA_original, BERTQA_memory23_future, BERTQA_memory23_question, BERTQA_future, BERTQA_memory_HAE_future, BERTQA_memoryHistory_future, BERTQA_question, BERTQA_memory_HAE_question, BERTQA_memoryHistory_question,BERTQA_memory23_switch
    
from transformers import BertTokenizer
from evaluate_utils import evaluate 
from extract_feature_bert_doqa import *
from utils import *

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

    wandb.init(project="doqa_twcc_eval_official", name=eval_config['exp_name'])
    wandb.config.update(config)

    config['device'] = torch.device("cuda")
    config['n_gpu']  = torch.cuda.device_count()

    model = model.to(config['device'])


    results = evaluate(config, eval_config['eval_file'], eval_config['eval_json'], model, tokenizer)

    record = {}
    for key, value in results.items():
        record["eval_{}".format(key)] = value

    print(record)

    wandb.log(record)

if __name__ == "__main__":
    main()
