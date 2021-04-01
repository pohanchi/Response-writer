import numpy as np
import IPython, pdb
import functools
import wandb
import json
import copy
from copy import deepcopy
from datasets import load_dataset, list_datasets, load_metric, Dataset
from transformers import BertModel, BertConfig, BertTokenizer
from torch import nn
from itertools import chain
import random
import re
import time
import string
import os
import pickle
import torch_optimizer as optim
import torch.nn.functional as F
from utils import logging
from enum import Enum
from multiprocessing import Pool, cpu_count
from functools import partial
from torch.nn.utils.rnn import pad_sequence 
from tqdm import trange
from collections import Counter
import spacy
import unicodedata
logger = logging

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def get_context_span(context, context_token):
    p_str = 0
    p_token = 0
    t_span = []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue

        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            log.info("Something wrong with get_context_span()")
            return []
        t_span.append((p_str, p_str + token_len))

        p_str += token_len
        p_token += 1
    return t_span

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def pre_proc(text):
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def find_answer_span(context_span, answer_start, answer_end):
    if answer_start == -1 and answer_end == -1:
        return (-1, -1)

    # s_1, t_1 = 0, 0
    t_start, t_end = 0, 0
    for token_id, (s, t) in enumerate(context_span):
        if s <= answer_start:
            t_start = token_id
        if t <= answer_end:
            t_end = token_id
    
    t_start = context_span[t_start][0]
    t_end = context_span[t_end][1]
        
    if t_start == -1 or t_end == -1:
        print(context_span, answer_start, answer_end)
        return (None, None)
    else:
        return (t_start, t_end)

def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))

def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', s)])

def free_text_to_span(free_text, full_text):
    if free_text == "unknown":
        return "unknown", -1, -1
    if normalize_answer(free_text) == "yes":
        return "yes", -1, -1
    if normalize_answer(free_text) == "no":
        return "no", -1, -1

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls)-1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i+j >= len(full_ls): break
            full_cnt[full_ls[i+j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0: continue

            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)

            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert(best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i+best_j][1]+1

    return full_text[char_i:char_j], char_i, char_j



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
    seq_length,
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
            max_seq_length=seq_length,
            doc_stride=doc_stride,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )

        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=16),
                total=len(examples),
                desc="convert examples to features",
                disable=not tqdm_enabled,
            )
        )

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
                all_input_ids, all_attention_masks, all_context_len, all_question_ids, \
                all_question_seg, all_question_attention_masks, all_question_len, all_question_start, all_dialog_act,\
                all_cls_index, all_p_mask, all_feature_index, \
                all_history_start, all_history_end,
            )
        else:

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

def convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, padding_strategy, is_training):
    
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

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_FIRST.value
        else:
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_FIRST.value
        

        encoded_dict = tokenizer.encode_plus(
            pairs,
            truncation=truncation,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - sequence_pair_added_tokens,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - sequence_pair_added_tokens, 
            )

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
            index = 1 + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = 1
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
            p_mask[0 :] = 0
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
                    doc_offset = 1

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        
        history_in_context =[]

        for index in range(len(history_span_modify)):
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            doc_offset = 1
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
        title: The title of the exampleconvert_examples_to_features
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
        end_position_character,
        answers=[],
        is_impossible=False,
        history_span_list=[],
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        self.dialog_act = dialog_act
        self.answers = answers
        self.history_origin = history_span_list
        # history_span_list = [ (answer_start, answer_end, ans_text) ]

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
            if self.history_origin[i][0] == -1:
                continue
            start_position_history = char_to_word_offset[self.history_origin[i][0]]
            end_position_history = char_to_word_offset[min(self.history_origin[i][1], len(char_to_word_offset)-1)]
            self.history_span_list.append([start_position_history, end_position_history, self.history_origin[i][2]])
        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and end_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[min(end_position_character, len(char_to_word_offset)-1)]

def convert_dataset_to_examples(datasets, mode):
    
    examples = []
    
    for index in trange(len(datasets[mode])):
        data = datasets[mode][index]

        start_position_character = None
        answer_text = None
        answers = []
        history_span_list = []

        if data["is_impossible"] == True and not data['answers'][0]['text'].lower() == "yes" and not data['answers'][0]['text'].lower() == "no":
            is_impossible = True
            answer = data['answers'][0]
            answer_text = answer['text'] 
        
        elif data['answers'][0]['text'].lower() == "yes" or data['answers'][0]['text'].lower() == "no":
            is_impossible = True
            answer = data['answers'][0]
            answer_text = answer['text']
            answers = data['answers']
        else:
            is_impossible = False
            answer = data['answers'][0]
            answer_text = answer['text']
            start_position_character = answer['answer_start']
            end_position_character = answer['answer_end']
            answers = data["answers"]

        num = data['id'].split("#")[-1]

        question = "[CLS] "
        if eval(num) != 0:
            previous = -eval(num)

            for i in range(previous,1,1):
                history = datasets[mode][index+i]
                if i !=0:
                    history_span_list.append([history['answers'][0]['answer_start'], history['answers'][0]['answer_end'], history['answers'][0]['text']])

                    question = question + pre_proc(history['question']) + " " + history['answers'][0]['gold_ans'] + " " +"[SEP]" + " "
                else:
                    question = question + pre_proc(history['question']) + " " +"[SEP]"
            
        else:
            question  = question + pre_proc(data['question']) + " [SEP]"

        dialog_act = data['dialog_act']

        example = RCExample(qas_id=data['id'],
                            question_text=question,
                            context_text=data["context"],
                            dialog_act=dialog_act,
                            answer_text=answer_text,
                            start_position_character=start_position_character, 
                            end_position_character=end_position_character,
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
        history_span_list=[]
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
            self.class_logits = cls_logits

def extract_and_save_feature(dataset_dict, mode, tokenizer, is_training, name):

    features, dataset = convert_examples_to_features(examples, tokenizer=tokenizer, doc_stride=stride,  is_training=is_training)

    torch.save(
    {"features": features, "dataset": dataset, "examples": examples},
    name,
    )



if __name__ == '__main__':

    mode = "train"
    name = "../dataset_local/coqa/coqa-train-v1.0.json"
    is_training = True
    is_dev = False
    dataset_raw = []
    max_seq_length = 384
    stride = 128
    ratio = 1
    cached_features_file = "../preprocessing_files/bert/CoQA/train_clean_truncated_add_hae"
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
    data = json.load(open(name))['data']
    dataset_dict = {mode:None}


    nlp = spacy.load("en", disable=['parser'])
    
    if os.path.exists(f"../dataset_local/coqa/dump_complete_{mode}.p"):
        print(f"load preload datset from ../dataset_local/coqa/dump_complete_{mode}.p")
        dataset_dict = pickle.load(open(f"../dataset_local/coqa/dump_complete_{mode}.p","rb"))
    else:
        print("file not exist, create a pickle file to preload in the next time ..")
        # preprocess data 
        for pa_pairs in tqdm(data):
            # remove redundant character to stablize extracting feature
            story = pa_pairs['story']
            context = [pre_proc(story)]
            doc = [context for context in nlp.pipe(context)][0]
            unnormed_tokens = [w.text for w in doc]

            for idx in range(len(pa_pairs['questions'])):

                data_pair = {}
                data_pair['question'] = pa_pairs['questions'][idx]['input_text']
                data_pair['id'] = pa_pairs['id'] + "_q#" + str(pa_pairs['questions'][idx]['turn_id']-1)
                data_pair['is_impossible'] = False

                if pa_pairs['answers'][idx]['span_text'] != "unknown":

                    if pa_pairs['answers'][idx]['input_text'].lower() == "yes" or pa_pairs['answers'][idx]['input_text'].lower() == "no":
                        data_pair['dialog_act'] = 1 if  pa_pairs['answers'][idx]['input_text'].lower() == "yes" else 2
                        data_pair['is_impossible'] = True
                        data_pair['answers'] = [{'text':pa_pairs['answers'][idx]['input_text'], 'answer_start':-1, 'answer_end':-1, 'gold_ans':pa_pairs['answers'][idx]['input_text'].lower()}]

                    else:
                        data_pair['dialog_act'] = 3
                        answer, char_i_span, char_j_span = free_text_to_span(pre_proc(pa_pairs['answers'][idx]['span_text']), context[0])
                        answer, char_i, char_j = free_text_to_span(pa_pairs['answers'][idx]['input_text'] ,answer)

                        ans_start =char_i_span + char_i
                        ans_end =char_j_span + char_j

                        data_pair['answers'] = [{'text':answer, 'answer_start':int(ans_start), 'answer_end':int(ans_end), 'gold_ans':pa_pairs['answers'][idx]['input_text']}]
                else:

                    data_pair['dialog_act'] = 0
                    data_pair['answers'] = [{'text':"unknown", 'answer_start': -1, 'answer_end':-1,'gold_ans':pa_pairs['answers'][idx]['input_text'].lower()}]
                    data_pair['is_impossible'] = True
                
                data_pair['context'] = context[0]
                dataset_raw.append(deepcopy(data_pair))
            dataset_dict[mode] = dataset_raw
            pickle.dump(dataset_dict, open(f"../dataset_local/coqa/dump_complete_{mode}.p","wb"))

    examples = convert_dataset_to_examples(dataset_dict,mode)

    if mode == "train":
        train_sample = ratio * len(examples)
        if not is_dev:
            examples = examples[:train_sample]
        else:
            examples = examples[train_sample:]

    features, dataset = convert_examples_to_features(examples, tokenizer=tokenizer, seq_length=max_seq_length, doc_stride=stride,  is_training=is_training, threads=8)

    torch.save(
    {"features": features, "dataset": dataset, "examples": examples},
    cached_features_file,
    )

