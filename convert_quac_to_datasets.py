from datasets import Dataset
import json

import IPython
import pdb

if __name__ == "__main__":
    mode = "train"
    name = "quac/train_v0.2.json"

    data = json.load(open(name))['data']

    dataset_dict = {"data":None}
    dataset = []
    for pa_pairs in data:
        context_pa_pair = pa_pairs['paragraphs'][0]
        context = context_pa_pair['context']
        for qa_pairs in context_pa_pair['qas']:
            qa_pairs.update({'context': context})
            dataset.append(qa_pairs)
    dataset_dict['data'] = dataset
    dataset_out=Dataset.from_dict(dataset_dict)
    IPython.embed()
    pdb.set_trace()
