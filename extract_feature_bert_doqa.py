import numpy as np
import IPython, pdb
import functools
import tqdm
import wandb
import json
import copy
from datasets import load_dataset, list_datasets, load_metric, Dataset
from transformers import BertModel, BertConfig, BertTokenizer
from torch import nn
from itertools import chain
import random
import re
import time
import torch_optimizer as optim
import torch.nn.functional as F
from utils import logging
from enum import Enum
from multiprocessing import Pool, cpu_count
from functools import partial
from torch.nn.utils.rnn import pad_sequence 
import argparse
import yaml

logger = logging

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`.
    Useful for tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"

def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def convert_examples_to_features(
    examples,
    tokenizer,
    doc_stride,
    is_training,
    padding_strategy="False",
    return_dataset='pt',
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi
    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`
    Example::
        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            convert_example_to_features,
            tokenizer=tokenizer,
            doc_stride=doc_stride,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )

        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=8),
                total=len(examples),
                desc="convert examples to features",
                disable=not tqdm_enabled,
            )
        )
    # features = convert_example_to_features(examples[2],tokenizer=tokenizer,doc_stride=doc_stride,padding_strategy=padding_strategy,is_training=is_training,)
    new_features = []
    unique_id = 1000000000
    example_index = 0


    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        # if not is_torch_available():
        #     raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features], batch_first=True)
        all_context_len = torch.tensor([len(f.input_ids) for f in features],dtype=torch.long)
        all_attention_masks = pad_sequence([torch.tensor(f.attention_mask, dtype=torch.long) for f in features],batch_first=True)
        all_question_ids = pad_sequence([torch.tensor(f.question_ans_ids, dtype=torch.long) for f in features],batch_first=True)
        all_question_seg = pad_sequence([torch.tensor(f.question_segs, dtype=torch.long) for f in features], batch_first=True)
        all_dialog_act = torch.tensor([f.dialog_act for f in features], dtype=torch.long)
        all_question_len = torch.tensor([f.question_len for f in features], dtype=torch.long)
        all_question_start = torch.tensor([f.question_start for f in features],dtype=torch.long)
        all_question_attention_masks = pad_sequence([torch.tensor(f.question_attention_masks, dtype=torch.long) for f in features], batch_first=True)
        all_cls_index = torch.tensor([f.cls_index for f in features],dtype=torch.long)
        all_p_mask = pad_sequence([torch.tensor(f.p_mask, dtype=torch.float) for f in features],batch_first=True)
        all_is_impossible = torch.tensor([f.is_impossible for f in features],dtype=torch.float)
        all_history_start = pad_sequence([torch.tensor(f.history_start_list, dtype=torch.long) for f in features], batch_first=True)
        all_history_end = pad_sequence([torch.tensor(f.history_end_list, dtype=torch.long) for f in features], batch_first=True)



        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, \
                all_context_len, all_question_ids, \
                all_question_seg, all_question_attention_masks, \
                all_question_len, all_question_start, \
                all_dialog_act, all_cls_index, \
                all_p_mask, all_feature_index, \
                all_history_start, all_history_end,
            )
        else:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in features],dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features],dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_context_len,
                all_question_ids,
                all_question_seg,
                all_question_attention_masks,
                all_question_len,
                all_question_start,
                all_dialog_act,
                all_start_positions,
                all_end_positions,
                all_feature_index,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
                all_history_start,
                all_history_end,
            )

        return features, dataset

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def convert_example_to_features(example, tokenizer, padding_strategy, is_training, doc_stride=128):
    
    features = []

    if is_training and not example.is_impossible:
        # Get start and end positionsequence_added_tokens
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    history_span_modify = []
    # handle history answer embedding
    for item in example.history_span_list:
        if item[0] == -1:
            continue
        history_start_position = orig_to_tok_index[item[0]]
        if item[1] < len(example.doc_tokens) - 1:
            history_end_position = orig_to_tok_index[item[1]+1] - 1
        else:
            history_end_position = len(all_doc_tokens) - 1
        
        (tok_history_start_position, tok_history_end_position) = _improve_answer_span(
            all_doc_tokens, history_start_position, history_end_position, tokenizer, item[2]
        )
        history_span_modify.append((tok_history_start_position, tok_history_end_position))        

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=False)

    indexes = np.where(np.array(truncated_query) == 102)
    question_attention_mask = np.ones_like(np.array(truncated_query)).tolist()

    if len(indexes[0])==1:

        question_seg = np.zeros_like(np.array(truncated_query))
        question_seg[1:indexes[0][-1]+1] = 1
        question_seg = question_seg.tolist()
        question_start = 1
        question_len = indexes[0][-1]+1

    else:
        indexes = indexes[0][::-1]
        question_seg = np.zeros_like(np.array(truncated_query))
        seg_value = [ 1, -1, -2 ]

        for i in range(len(indexes)-1):
            if i < 3:
                if i+1 == (len(indexes)-1):
                    question_seg[1:indexes[i]+1] = seg_value[i]
                else:
                    question_seg[indexes[i+1]+1:indexes[i]+1] = seg_value[i]
            else:
                if i+1 == (len(indexes)-1):
                    question_seg[1:indexes[i]+1] = 0
                else:
                    question_seg[indexes[i+1]:indexes[i]+1] = 0
            if i == 0:
                question_start = indexes[i+1]+1
                question_len = indexes[0]+1

        question_seg = question_seg.tolist()

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
    sequence_added_tokens = 1

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            pairs = span_doc_tokens
            truncation = TruncationStrategy.DO_NOT_TRUNCATE.value
        else:
            pairs = span_doc_tokens
            truncation = TruncationStrategy.DO_NOT_TRUNCATE.value
        

        encoded_dict = tokenizer.encode_plus(
            pairs,
            truncation=truncation,
            return_token_type_ids=True
        )

        paragraph_len = len(all_doc_tokens)

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = i+sequence_added_tokens #(different! index = i+sequence_added_tokens)
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = sequence_added_tokens # (different! encoded_dict["truncated_query_with_special_tokens_length"] = sequence_added_tokens)
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:

        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[sequence_added_tokens :] = 0  #(different than offficial code (p_mask[sequence_added_tokens :] = 0))
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)

        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = sequence_added_tokens # different! ( doc_offset = sequence_added_tokens)

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        
        history_in_context =[]
        for index in range(len(history_span_modify)):
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            doc_offset = sequence_added_tokens
            if not history_span_modify[index][0] >= doc_start and history_span_modify[index][1] <= doc_end:
                continue
            else:
                start_history_position = history_span_modify[index][0] - doc_start + doc_offset
                end_history_position = history_span_modify[index][1] - doc_start + doc_offset
                history_in_context.append((start_history_position, end_history_position))
        
        features.append(
            RCFeatures(
                span["input_ids"],
                span["attention_mask"],
                truncated_query,
                question_seg,
                question_attention_mask,
                question_start,
                question_len,
                example.dialog_act,
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
                history_span_list=history_in_context,
            )
        )
    return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class RCExample:
    """
    A single training/test example for the ReadingComprehension dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        dialog_act,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
        history_span_list=[],
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.dialog_act = dialog_act
        self.answers = answers
        self.history_origin = history_span_list
        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset
        self.history_span_list = []
        for i in range(len(self.history_origin)):
            if self.history_origin[i][2] == "CANNOTANSWER":
                start_position_history = -1
                end_position_history = -1
            else:
                start_position_history = char_to_word_offset[self.history_origin[i][0]]
                end_position_history = char_to_word_offset[min(self.history_origin[i][0] + len(self.history_origin[i][2]) - 1, len(char_to_word_offset)-1)]
            self.history_span_list.append([start_position_history, end_position_history, self.history_origin[i][2]])
        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

def convert_dataset_to_examples(datasets, mode):
    
    examples = []
    
    for index in range(len(datasets[mode])):
        data = datasets[mode][index]

        start_position_character = None
        answer_text = None

        is_impossible = True if data['orig_answer']['text'][0] =="CANNOTANSWER" else False
        answer_text = data['orig_answer']['text'][0] if not is_impossible else "CANNOTANSWER"
        start_position_character = data['orig_answer']['answer_start'][0]
        answers = data["answers"]

        num = data['id'].split("#")[-1]

        question = "[CLS] "

        history_span_list = []

        if eval(num) != 0:
            previous = -eval(num)
            
            for i in range(previous,1,1):
                history = datasets[mode][index+i]
                if i !=0:
                    history_span_list.append([history['orig_answer']['answer_start'][0], None, history['orig_answer']['text'][0]])
                    question = question + history['question'] + " " + history['orig_answer']['text'][0] + " " +"[SEP]" + " "
                else:
                    question = question + history['question'] + " " +"[SEP]"


            dialog_act = 1 if datasets[mode][index-1]['followup'] == 'y' else 0
        else:
            question  = question + data['question'] + " [SEP]"
            dialog_act = 1

        example = RCExample(qas_id=data['id'],
                            question_text=question,
                            context_text=data["context"],
                            dialog_act=dialog_act,
                            answer_text=answer_text,
                            start_position_character=start_position_character, 
                            title=data['title'],
                            is_impossible=is_impossible,
                            answers=answers,
                            history_span_list=history_span_list)
    
        examples.append(example)

    return examples

class RCFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        question_ans_ids,
        question_segs,
        question_attention_masks,
        question_start,
        question_len,
        dialog_act,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
        history_span_list=[],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.question_ans_ids = question_ans_ids
        self.question_segs = question_segs
        self.question_attention_masks = question_attention_masks
        self.question_len = question_len
        self.question_start = question_start
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.dialog_act = dialog_act
        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.history_start_list = []
        self.history_end_list = []
        
        for item in history_span_list:
            self.history_start_list.append(item[0])
            self.history_end_list.append(item[1])
        


class RCResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits




def extract_and_save_feature(dataset, mode, tokenizer, is_training, name, ratio, is_dev, stride=128):
    
    examples = convert_dataset_to_examples(dataset,mode)

    if mode == "train":
        train_sample = int(ratio * len(examples))
        if not is_dev:
            examples = examples[:train_sample]
        else:
            examples = examples[train_sample:]

    features, dataset = convert_examples_to_features(examples, tokenizer=tokenizer, doc_stride=stride,  is_training=is_training)

    torch.save(
    {"features": features, "dataset": dataset, "examples": examples},
    name,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argument Parser for HistoryQA project.')
    parser.add_argument("--config")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config,"r"))

    dataset = load_dataset("doqa", config['domain'],cache_dir="../dataset_local/doqa")
    cached_features_file = config['output_name']
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    extract_and_save_feature(dataset, config['mode'], tokenizer, config['is_training'], cached_features_file, ratio=config['ratio'], is_dev=config['is_dev'])
