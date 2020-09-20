import numpy
import torch
import IPython, pdb
from datasets import load_dataset, list_datasets, load_metric
from transformers import BertModel, BertConfig, BertTokenizer
from module_custom import TransformerDecoder, TransformerDecoderLayer, TransformerDecoderLayer_bmask

import logging
logging.basicConfig(level=logging.INFO)


def example_utilize_BERT():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs[0]
    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
    assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
    return

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    squad_dataset = load_dataset("natural_questions", cache_dir="./nq")
    
    squad_metric = load_metric('natural_questions')

    special_tokens_dict = {"additional_special_tokens":['<Q>','<ans>','<eos>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    model.resize_token_embeddings(len(tokenizer))

    IPython.embed()  
    pdb.set_trace()