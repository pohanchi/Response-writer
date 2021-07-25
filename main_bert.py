import numpy as np
import torch
import argparse
import yaml
import wandb
import transformers

os.environ['WANDB_MODE'] = 'dryrun'

def main():
    parser = argparse.ArgumentParser(description="Argument Parser for HistoryQA project")
    parser.add_argument("-c", "--config", description="Config Path")
    parser.add_argument("-n", "--name", description="Experiment Name")
    parser.add_argument("-k", "--kind", description="Different kind of Model [ 'FlowQA', 'FlowDelta', 'BERT', 'PHBERT', 'HisBERT', 'HistoryModule' ]")
    parser.add_argument("-s", "--seed", type=int, default=42, description="Random Seed")
    parser.add_argument("-d", "--dataset", description="Dataset Name")
    parser.add_argument("-g", "--n_gpus", description="N GPUs")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    set_seed(args.seed)
    




if __name__ == "__main__":
    main()