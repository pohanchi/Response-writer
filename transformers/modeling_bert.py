# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import copy
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import Linear, Dropout

from .activations import ACT2FN
from .configuration_bert import BertConfig
from .file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    apply_chunking_to_forward_memory,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from .utils import logging
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import weight_norm
import IPython
import pdb
import torch.nn.functional as F

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class BertEmbeddingsRelative(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            if seq_length <= 512:
                position_ids = self.position_ids[:, :seq_length]
            else:
                position_ids = torch.arange(seq_length).expand((1, -1)).cuda()
                position_ids = position_ids % 512

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddingsMemory2(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, memory_ids=None, memory_embed=None, memory=False):
        
        if memory == True:
        
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            if position_ids is None:
                if seq_length <= 512:
                    position_ids = self.position_ids[:, :seq_length]
                else:
                    position_ids = torch.arange(seq_length).expand((1, -1)).cuda()
                    position_ids = position_ids % 512

            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = memory_embed
            embeddings = self.LayerNorm(embeddings + inputs_embeds)
            embeddings = self.dropout(embeddings)
            return embeddings
        
        else:
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            if position_ids is None:
                if seq_length <= 512:
                    position_ids = self.position_ids[:, :seq_length]
                else:
                    position_ids = torch.arange(seq_length).expand((1, -1)).cuda()
                    position_ids = position_ids % 512

            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + token_type_embeddings + position_embeddings #+ memory_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
            
            return embeddings


class BertEmbeddingsMemoryMarker(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.marker_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, memory_ids=None, memory_embed=None):
                
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            if seq_length <= 512:
                position_ids = self.position_ids[:, :seq_length]
            else:
                position_ids = torch.arange(seq_length).expand((1, -1)).cuda()
                position_ids = position_ids % 512

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        marker_embeddings =self.marker_embeddings(memory_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings + marker_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


def build_relative_position(query_size, key_size, device):
    """ Build relative position according to the query and key
    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key :math:`P_k` is range from (0, key_size),
    The relative positions from query to key is
    
    :math:`R_{q \\rightarrow k} = P_q - P_k`
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """

    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class BertSelfAttentionRelative(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'none').lower().split('|')] # c2p|p2c
        self.relative_attention = getattr(config, 'relative_attention', False)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions <1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = Dropout(config.hidden_dropout_prob)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(query_layer.size(-1)*scale_factor)        
        query_layer = query_layer / scale

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.relative_attention:
            relative_pos = None
            rel_embedding = rel_embeddings(torch.arange(0,self.max_relative_positions*2).cuda())
            rel_embedding = self.pos_dropout(rel_embedding)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embedding, scale_factor)      

        if self.relative_attention:
            attention_scores = (attention_scores + rel_att)

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span, :].unsqueeze(0)


        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, 2*att_span-1)

            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att
        

        return score





class BertSelfAttentionMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'none').lower().split('|')] # c2p|p2c
        self.relative_attention = getattr(config, 'relative_attention', False)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions <1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = Dropout(config.hidden_dropout_prob)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embeddings,
        mem,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        hidden_states_tmp = hidden_states
        mixed_query_layer = self.query(hidden_states_tmp)
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            # import ipdb; ipdb.set_trace()

            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)

            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states_tmp)
            mixed_value_layer = self.value(hidden_states_tmp)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(query_layer.size(-1)*scale_factor)        
        query_layer = query_layer / scale


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        if self.relative_attention:
            relative_pos = None
            rel_embedding = rel_embeddings(torch.arange(0,self.max_relative_positions*2).cuda())
            rel_embedding = self.pos_dropout(rel_embedding)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embedding, scale_factor)     


        if self.relative_attention:
            attention_scores = (attention_scores + rel_att)

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span, :].unsqueeze(0)


        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, 2*att_span-1)

            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att
        

        return score

class BertSelfAttentionReuse(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'none').lower().split('|')] # c2p|p2c
        self.relative_attention = getattr(config, 'relative_attention', False)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)


        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions <1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = Dropout(config.hidden_dropout_prob)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embeddings,
        mem,
        attention_mask=None,
        mem_att_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        history_module=False
    ):
        

        if not history_module:

            hidden_states_tmp = hidden_states
            mixed_query_layer = self.query(hidden_states_tmp)
            
            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder; the attention mask needs to be
            # such that the encoder's padding tokens are not attended to.
            if encoder_hidden_states is not None:

                mixed_key_layer = self.key(encoder_hidden_states)
                mixed_value_layer = self.value(encoder_hidden_states)

                attention_mask = encoder_attention_mask
            else:
                mixed_key_layer = self.key(hidden_states_tmp)
                mixed_value_layer = self.value(hidden_states_tmp)


            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.

            scale_factor = 1
            if 'c2p' in self.pos_att_type:
                scale_factor += 1
            if 'p2c' in self.pos_att_type:
                scale_factor += 1
            if 'p2p' in self.pos_att_type:
                scale_factor += 1

            scale = math.sqrt(query_layer.size(-1)*scale_factor)        
            query_layer = query_layer / scale


            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


            if self.relative_attention:
                relative_pos = None
                rel_embedding = rel_embeddings(torch.arange(0,self.max_relative_positions*2).cuda())
                rel_embedding = self.pos_dropout(rel_embedding)
                rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embedding, scale_factor)     


            if self.relative_attention:
                attention_scores = (attention_scores + rel_att)

            # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # import ipdb; ipdb.set_trace()
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs
        
        else:

            hidden_states_tmp = hidden_states
            mixed_query_layer = self.query(hidden_states_tmp)
            
            mixed_key_layer = self.key(hidden_states_tmp)
            mixed_value_layer = self.value(hidden_states_tmp)

            mixed_mem_query = self.query(mem)


            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            history_query_layer = self.transpose_for_scores(mixed_mem_query)

            # Take the dot product between "query" and "key" to get the raw attention scores.

            scale_factor = 1
            if 'c2p' in self.pos_att_type:
                scale_factor += 1
            if 'p2c' in self.pos_att_type:
                scale_factor += 1
            if 'p2p' in self.pos_att_type:
                scale_factor += 1

            scale = math.sqrt(query_layer.size(-1)*scale_factor)        
            query_layer = query_layer / scale
            history_query_layer = history_query_layer / scale


            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_bias_scores = torch.matmul(history_query_layer, key_layer.transpose(-1, -2))
            attention_bias_scores = attention_bias_scores[:,:,0][:,:,None]

            if self.relative_attention:
                relative_pos = None
                rel_embedding = rel_embeddings(torch.arange(0,self.max_relative_positions*2).cuda())
                rel_embedding = self.pos_dropout(rel_embedding)
                rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embedding, scale_factor)     


            if self.relative_attention:
                attention_scores = (attention_scores + rel_att)

            # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # import ipdb; ipdb.set_trace()
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask + attention_bias_scores

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs

    
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span, :].unsqueeze(0)


        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, 2*att_span-1)

            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att
        

        return score




class BertSelfAttentionMemoryweight(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'none').lower().split('|')] # c2p|p2c
        self.relative_attention = getattr(config, 'relative_attention', False)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions <1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = Dropout(config.hidden_dropout_prob)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embeddings,
        mem,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        hidden_states_tmp = hidden_states
        mixed_query_layer = self.query(hidden_states_tmp)        

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)

            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states_tmp)
            mixed_value_layer = self.value(hidden_states_tmp)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1

        scale = math.sqrt(query_layer.size(-1)*scale_factor)        
        query_layer = query_layer / scale


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        if self.relative_attention:
            relative_pos = None
            rel_embedding = rel_embeddings(torch.arange(0,self.max_relative_positions*2).cuda())
            rel_embedding = self.pos_dropout(rel_embedding)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embedding, scale_factor)     


        if self.relative_attention:
            attention_scores = (attention_scores + rel_att)

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span, :].unsqueeze(0)


        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, 2*att_span-1)

            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att
        

        return score

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfOutputMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertAttentionRelative(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttentionRelative(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        relative_embedding,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            relative_embedding,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertAttentionMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttentionMemory(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        mem,
        relative_embedding,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        cross_att=False,
    ):

        if cross_att == False:
            encoder_attention_mask= None
            encoder_hidden_states = None


        self_outputs = self.self(
            hidden_states,
            relative_embedding,
            mem,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output, ) + self_outputs[2:]  # add attentions if we output them
        return outputs

class BertAttentionMemoryweight(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttentionMemoryweight(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        mem,
        relative_embedding,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            relative_embedding,
            mem,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output, ) + self_outputs[2:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertOutputMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm((hidden_states + input_tensor))

        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLayerRelative(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionRelative(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttentionRelative(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        relative_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            relative_embeddings,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLayerMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionMemory(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttentionMemory(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutputMemory(config)


    def forward(
        self,
        hidden_states,
        mem,
        relative_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        cross_att=False,
        not_self_attention=False,
    ):
        if not not_self_attention:
            self_attention_outputs = self.attention(
                hidden_states,
                mem,
                relative_embeddings,
                attention_mask,
                head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                cross_att=cross_att,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[2:]
        else:
            attention_output = hidden_states
            outputs = ( 0, 0 )

        #attention_output = self_attention_outputs[0]

        #outputs = self_attention_outputs[2:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_attention_mask,
                output_attentions,
            )

            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward_memory(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)

        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class BertEncoderRelative(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayerRelative(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        relative_embeddings,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    relative_embeddings,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    relative_embeddings,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class BertEncoderHisBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayerMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        relative_embeddings,
        memory,
        memory_module=None,
        memory_len=None,
        input_len=None,
        attention_mask=None,
        memory_att_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        index=12,
        start=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if i < start:
                continue 
            if i >= index:
                break

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    memory,
                    relative_embeddings,
                    memory_att_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:

                if i == 11 or i == 1:
                    layer_outputs = layer_module(
                        hidden_states,
                        memory,
                        relative_embeddings,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                        cross_att=True,
                    )
                else:

                    layer_outputs = layer_module(
                        hidden_states,
                        memory,
                        relative_embeddings,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        output_attentions,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, memory, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class BertEncoderMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayerMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        relative_embeddings,
        memory,
        memory_module=None,
        memory_len=None,
        input_len=None,
        attention_mask=None,
        memory_att_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        index=12,
        start=0,
        not_self_attention=False,
        cross_att=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if i < start:
                continue 
            if i >= index:
                break

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    memory,
                    relative_embeddings,
                    memory_att_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    not_self_attention=not_self_attention,
                    cross_att=cross_att,
                )
            else:

                
                layer_outputs = layer_module(
                    hidden_states,
                    memory,
                    relative_embeddings,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    not_self_attention=not_self_attention,
                    cross_att=cross_att,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, memory, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModelRelative(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsRelative(config)
        self.encoder = BertEncoderRelative(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            self.embeddings.position_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModelRelativeHAE(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemoryMarker(config)
        self.encoder = BertEncoderRelative(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        history_ids=None,
        **kwargs,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, memory_ids=history_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            self.embeddings.position_embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class BertInjectMemory(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.memory_query= nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value = nn.Linear(config.hidden_size, value_size)
        self.memory_key = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.g = nn.Parameter(torch.randn(config.hidden_size,1))
        self.activation=ACT2FN['gelu']
        
        # self.memory_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.m_ = nn.Softmax(dim=0)
        self.m2 = nn.Softmax(dim=1)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index,transform=None):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, query=self.memory_query,key=self.memory_key,value=self.memory_value,transform=transform)
        # hidden_states_tmp2 = self.memory_LayerNorm(hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp1, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index,query=None, key=None, value=None, transform=None):
        

        memory_global_matrix = []
        for i in range(len(hidden_states)):
            if value is not None and key is not None:
                memory_1d = memory[i][:history_len[i]]
                lambda_layer=torch.matmul(self.m_(key(memory_1d).transpose(1,0)),value(memory_1d))
                lambda_layer=self.memory_project(self.activation(lambda_layer))
            elif key is not None:
                lambda_layer=torch.matmul(key(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])
            
            elif value is not None:
                lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),value(memory[i][:history_len[i]]))
            
            else:
                lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])

            memory_global_matrix.append(lambda_layer)
        global_memory_lambda_matrix = torch.stack(memory_global_matrix)

        if query is not None:
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1] 
            hidden_size  = hidden_states.shape[2]
            # global information
            global_memory_lambda_matrix_a = torch._weight_norm(global_memory_lambda_matrix, self.g,-1)
            
            # local information
            # local_memory_lambda_matrix_b = torch._weight_norm(local_memory_lambda_matrix,self.h, -2)
            
            query_vector = query[index](hidden_states)
            hidden_states_tmp = self.activation(query_vector)
            output = torch.bmm(hidden_states_tmp, global_memory_lambda_matrix_a)
            
            return output, global_memory_lambda_matrix_a
        else:
            output = torch.bmm( hidden_states * self.m2(query[index](hidden_states)), global_memory_lambda_matrix)
            return output, global_memory_lambda_matrix


class BertHistoryGenerator(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.memory_query= nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_key = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_project = nn.Linear(config.hidden_size, 2)
        self.memory_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.m_ = nn.Softmax(dim=0)
        self.m2 = nn.Softmax(dim=1)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, last=False):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index)
        hidden_states_tmp2 = self.memory_LayerNorm(hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index):
        

        memory_global_matrix = []
        all_probs = []
        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            lambda_layer=torch.matmul(self.m_(self.memory_key(memory_1d)).transpose(1,0),self.memory_value(memory_1d))
            memory_global_matrix.append(self.memory_project(lambda_layer))
            input_1d = hidden_states[i][:input_len[i]]
        global_memory_lambda_matrix = torch.stack(memory_global_matrix)

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        output = (self.memory_query[index](hidden_states) * global_memory_lambda_matrix[:,:,0][:,None]) + global_memory_lambda_matrix[:,:,1][:,None]


        return output, global_memory_lambda_matrix




class BertHistoryGenerator9(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.memory_query= nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_key = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_project = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(1)])
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(1)])


        self.memory_query_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_key_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_project2 = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(1)])               
        self.m_ = nn.Softmax(dim=0)
        self.m2 = nn.Softmax(dim=1)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, last=False):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index)
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, hidden_states_tmp1


    def memory_operator(self, hidden_states, memory, history_len, input_len, index):
        

        memory_global_matrix = []
        memory_vertical_matrix = []
        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 64 else 64, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]

            lambda_layer=torch.matmul(self.m_(self.memory_key[index](memory_1d)).transpose(1,0),self.memory_value[index](memory_1d))
            memory_global_matrix.append(self.memory_project[index](lambda_layer))

            lambda_layer_seqlen = torch.matmul(self.memory_key_vertical[index](hidden_states[i]), self.m_(self.memory_value_vertical[index](memory_1d)).transpose(1,0))
            temporal_matrix = self.memory_project2[index](memory_1d)
            matrix_w_bias = torch.matmul(lambda_layer_seqlen, temporal_matrix)
            input_1d = hidden_states[i][:input_len[i]]   
            memory_vertical_matrix.append(matrix_w_bias)         



        global_memory_lambda_matrix = torch.stack(memory_global_matrix)
        global_memory_vertical_matrix = torch.stack(memory_vertical_matrix)


        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        output1 = self.memory_query[index](hidden_states) * global_memory_lambda_matrix[:,:,0][:,None] + global_memory_lambda_matrix[:,:,1][:,None]
        output2 = self.memory_query_vertical[index](hidden_states) * global_memory_vertical_matrix[:,:,0][:,:,None] + global_memory_vertical_matrix[:,:,1][:,:,None]

        output = output1 + output2

        return output, global_memory_lambda_matrix


class BertHistoryGenerator22(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.memory_query = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_key = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_project = nn.ModuleList([nn.Linear(config.hidden_size, 1*2) for _ in range(1)])
        self.memory_hidden = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_hidden2 = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])

        self.memory_query_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_key_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_project2 = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size * 2) for _ in range(1)])               
        self.m_ = nn.Softmax(dim=0)
        self.m2 = nn.Softmax(dim=1)
        self.activation = ACT2FN['gelu']



    
    def forward(self,hidden_states, memory, memory_len, input_len, index, last=False):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index)
        return hidden_states_tmp1, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index):
        

        memory_global_matrix = []
        memory_vertical_matrix = []
        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]
            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 64 else 64, sorted=True)
            memory_1d = memory_1d[selected_index]
            memory_1d = self.activation(self.memory_hidden2[index](self.activation(self.memory_hidden[index](memory_1d))))
            lambda_layer=self.m_(torch.matmul(self.memory_key[index](memory_1d).transpose(1,0),self.memory_value[index](memory_1d)) / torch.sqrt(torch.FloatTensor([memory.shape[-2]]).to(hidden_states.device)))
            memory_global_matrix.append(self.memory_project[index](lambda_layer))

            lambda_layer_seqlen = torch.matmul(self.memory_key_vertical[index](hidden_states[i]), self.memory_value_vertical[index](memory_1d).transpose(1,0))
            temporal_matrix = self.memory_project2[index](memory_1d)
            matrix_w_bias = torch.matmul(self.m2(lambda_layer_seqlen/torch.sqrt(torch.FloatTensor([hidden_states.shape[-1]]).to(hidden_states.device))), temporal_matrix)

            memory_vertical_matrix.append(matrix_w_bias)         

        global_memory_lambda_matrix = torch.stack(memory_global_matrix)
        global_memory_vertical_matrix = torch.stack(memory_vertical_matrix)
        global_memory_vertical_matrix1 = global_memory_vertical_matrix[:,:,:hidden_states.shape[-1]]
        global_memory_vertical_matrix2 = global_memory_vertical_matrix[:,:,hidden_states.shape[-1]:]

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]
        output1 = hidden_states * global_memory_vertical_matrix1 + global_memory_lambda_matrix[:,:,0][:,None]
        hid = self.activation(output1)
        output2 = hid * global_memory_vertical_matrix2 + global_memory_lambda_matrix[:,:,1][:,None]

        output = output2
        


        return output, global_memory_lambda_matrix


class BertHistoryGenerator10(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.memory_query= nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_key = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_project = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(1)])
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(1)])


        self.memory_query_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_value_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_key_vertical = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(1)])
        self.memory_project2 = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(1)])               
        self.m_ = nn.Softmax(dim=0)
        self.m2 = nn.Softmax(dim=1)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, last=False):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index)
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index):
        

        memory_global_matrix = []
        memory_vertical_matrix = []
        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 64 else 64, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]

            lambda_layer=torch.matmul(self.m_(self.memory_key[index](memory_1d)).transpose(1,0),self.memory_value[index](memory_1d))
            memory_global_matrix.append(self.memory_project[index](lambda_layer))

            lambda_layer_seqlen = torch.matmul(self.memory_key_vertical[index](hidden_states[i]), self.m_(self.memory_value_vertical[index](memory_1d)).transpose(1,0))
            temporal_matrix = self.memory_project2[index](memory_1d)
            matrix_w_bias = torch.matmul(lambda_layer_seqlen, temporal_matrix)
            input_1d = hidden_states[i][:input_len[i]]   
            memory_vertical_matrix.append(matrix_w_bias)         



        global_memory_lambda_matrix = torch.stack(memory_global_matrix)
        global_memory_vertical_matrix = torch.stack(memory_vertical_matrix)


        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        output1 = self.memory_query[index](hidden_states) * global_memory_lambda_matrix[:,:,0][:,None] + global_memory_lambda_matrix[:,:,1][:,None]
        output2 = self.memory_query_vertical[index](hidden_states) * global_memory_vertical_matrix[:,:,0][:,:,None] + global_memory_vertical_matrix[:,:,1][:,:,None]

        output = output1 + output2

        return output, global_memory_lambda_matrix




class BertHistoryGenerator3(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.m_ = nn.Softmax(dim=0)
        self.act_fn = ACT2FN['gelu']
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(13)])


        self.memory_qq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_qv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_qk = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.qbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.qlq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.memory_vq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_vv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_vk = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.vbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.vlq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.memory_kq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_kv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_kk = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.kbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.klq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.memory_linq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_linv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_link = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.linbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.linlq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.m2 = nn.Softmax(dim=1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, input_att_mask):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, input_att_mask)
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index, input_att_mask=None):
        

        memory_global_matrix = []
        memory_bias_matrix = []
        
        memory_global_matrix_V = []
        memory_bias_matrix_V = []    

        memory_global_matrix_K = []
        memory_bias_matrix_K = []     

        memory_global_matrix_linear = []
        memory_bias_matrix_linear = []  

        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            lambda_layer=torch.matmul(self.m_(self.memory_qk[index](memory_1d)).transpose(1,0),self.memory_qv[index](memory_1d))
            lambda_bias = torch.matmul(self.m_(self.qbk[index](memory_1d)).transpose(1,0), self.memory_qv[index](memory_1d))
            memory_bias_matrix.append(lambda_bias)
            memory_global_matrix.append(lambda_layer)

            lambda_layer_V=torch.matmul(self.m_(self.memory_vk[index](memory_1d)).transpose(1,0),self.memory_vv[index](memory_1d))
            lambda_bias_V = torch.matmul(self.m_(self.vbk[index](memory_1d)).transpose(1,0), self.memory_vv[index](memory_1d))
            memory_bias_matrix_V.append(lambda_bias_V)
            memory_global_matrix_V.append(lambda_layer_V)

            lambda_layer_K=torch.matmul(self.m_(self.memory_kk[index](memory_1d)).transpose(1,0),self.memory_kv[index](memory_1d))
            lambda_bias_K = torch.matmul(self.m_(self.kbk[index](memory_1d)).transpose(1,0), self.memory_kv[index](memory_1d))
            memory_bias_matrix_K.append(lambda_bias_K)
            memory_global_matrix_K.append(lambda_layer_K)

            lambda_layer_linear=torch.matmul(self.m_(self.memory_link[index](memory_1d)).transpose(1,0),self.memory_linv[index](memory_1d))
            lambda_bias_linear = torch.matmul(self.m_(self.linbk[index](memory_1d)).transpose(1,0), self.memory_linv[index](memory_1d))
            memory_bias_matrix_linear.append(lambda_bias_linear)
            memory_global_matrix_linear.append(lambda_layer_linear)


        global_memory_lambda_matrix = torch.stack(memory_global_matrix)
        global_memory_lambda_matrix_bias = torch.stack(memory_bias_matrix) 

        global_memory_lambda_matrix_V = torch.stack(memory_global_matrix_V)
        global_memory_lambda_matrix_bias_V = torch.stack(memory_bias_matrix_V)

        global_memory_lambda_matrix_K = torch.stack(memory_global_matrix_K)
        global_memory_lambda_matrix_bias_K = torch.stack(memory_bias_matrix_K)

        global_memory_lambda_matrix_linear = torch.stack(memory_global_matrix_linear)
        global_memory_lambda_matrix_bias_linear = torch.stack(memory_bias_matrix_linear)
        

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        #output_q = torch.bmm(self.memory_qq[index](hidden_states), global_memory_lambda_matrix) + global_memory_lambda_matrix_bias
        output_v = torch.bmm(self.memory_vq[index](hidden_states), global_memory_lambda_matrix_V) + global_memory_lambda_matrix_bias_V
        #output_k = torch.bmm(self.memory_kq[index](hidden_states), global_memory_lambda_matrix_K) + global_memory_lambda_matrix_bias_K

        q_layer = self.transpose_for_scores(hidden_states)
        v_layer = self.transpose_for_scores(output_v)
        k_layer = self.transpose_for_scores(hidden_states)

        attention_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if input_att_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + input_att_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output_context = torch.bmm(self.memory_linq[index](context_layer), global_memory_lambda_matrix_linear) + global_memory_lambda_matrix_bias_linear
        
        return output_context, attention_probs


class BertHistoryGenerator4(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.m_ = nn.Softmax(dim=0)
        self.act_fn = ACT2FN['gelu']
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(13)])


        self.memory_qq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_qv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_qk = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.qbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.qlq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.memory_vq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_vv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_vk = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.vbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.vlq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.memory_kq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_kv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_kk = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.kbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.klq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.memory_linq= nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.memory_linv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_link = nn.ModuleList([nn.Linear(config.hidden_size, 32) for _ in range(13)])
        self.linbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        self.linlq = nn.ModuleList([nn.Linear(32, 32) for _ in range(13)])

        self.m2 = nn.Softmax(dim=1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, input_att_mask):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, input_att_mask)
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index, input_att_mask=None):
        

        memory_global_matrix = []
        memory_bias_matrix = []
        
        memory_global_matrix_V = []
        memory_bias_matrix_V = []    

        memory_global_matrix_K = []
        memory_bias_matrix_K = []     

        memory_global_matrix_linear = []
        memory_bias_matrix_linear = []  

        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 32 else 32, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]
            lambda_layer=torch.matmul(self.m_(self.memory_qk[index](memory_1d)).transpose(1,0),self.memory_qv[index](memory_1d))
            lambda_bias = torch.matmul(self.m_(self.qbk[index](memory_1d)).transpose(1,0), self.memory_qv[index](memory_1d))
            memory_bias_matrix.append(lambda_bias)
            memory_global_matrix.append(lambda_layer)

            lambda_layer_V=torch.matmul(self.m_(self.memory_vk[index](memory_1d)).transpose(1,0),self.memory_vv[index](memory_1d))
            lambda_bias_V = torch.matmul(self.m_(self.vbk[index](memory_1d)).transpose(1,0), self.memory_vv[index](memory_1d))
            memory_bias_matrix_V.append(lambda_bias_V)
            memory_global_matrix_V.append(lambda_layer_V)

            lambda_layer_K=torch.matmul(self.m_(self.memory_kk[index](memory_1d)).transpose(1,0),self.memory_kv[index](memory_1d))
            lambda_bias_K = torch.matmul(self.m_(self.kbk[index](memory_1d)).transpose(1,0), self.memory_kv[index](memory_1d))
            memory_bias_matrix_K.append(lambda_bias_K)
            memory_global_matrix_K.append(lambda_layer_K)

            lambda_layer_linear=torch.matmul(self.m_(self.memory_link[index](memory_1d)).transpose(1,0),self.memory_linv[index](memory_1d))
            lambda_bias_linear = torch.matmul(self.m_(self.linbk[index](memory_1d)).transpose(1,0), self.memory_linv[index](memory_1d))
            memory_bias_matrix_linear.append(lambda_bias_linear)
            memory_global_matrix_linear.append(lambda_layer_linear)


        global_memory_lambda_matrix = torch.stack(memory_global_matrix)
        global_memory_lambda_matrix_bias = torch.stack(memory_bias_matrix) 

        global_memory_lambda_matrix_V = torch.stack(memory_global_matrix_V)
        global_memory_lambda_matrix_bias_V = torch.stack(memory_bias_matrix_V)

        global_memory_lambda_matrix_K = torch.stack(memory_global_matrix_K)
        global_memory_lambda_matrix_bias_K = torch.stack(memory_bias_matrix_K)

        global_memory_lambda_matrix_linear = torch.stack(memory_global_matrix_linear)
        global_memory_lambda_matrix_bias_linear = torch.stack(memory_bias_matrix_linear)
        

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        # import ipdb; ipdb.set_trace()

        #output_q = torch.bmm(self.memory_qq[index](hidden_states), global_memory_lambda_matrix) + global_memory_lambda_matrix_bias
        output_v = torch.bmm(self.memory_vq[index](hidden_states), global_memory_lambda_matrix_V) + global_memory_lambda_matrix_bias_V
        #output_k = torch.bmm(self.memory_kq[index](hidden_states), global_memory_lambda_matrix_K) + global_memory_lambda_matrix_bias_K

        q_layer = self.transpose_for_scores(hidden_states)
        v_layer = self.transpose_for_scores(output_v)
        k_layer = self.transpose_for_scores(hidden_states)

        attention_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if input_att_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + input_att_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output_context = torch.bmm(self.memory_linq[index](context_layer), global_memory_lambda_matrix_linear) + global_memory_lambda_matrix_bias_linear
        
        return output_context, attention_probs


class BertHistoryGenerator5(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.m_ = nn.Softmax(dim=0)
        self.act_fn = ACT2FN['gelu']
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(13)])

        self.memory_linq= nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_linv = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_link = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(13)])
        self.linbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])

        self.m2 = nn.Softmax(dim=1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, input_att_mask):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, input_att_mask)
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index, input_att_mask=None):  

        memory_global_matrix_linear = []
        memory_bias_matrix_linear = []  

        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 32 else 32, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]

            lambda_layer_linear=torch.matmul(self.memory_link[index](memory_1d).transpose(1,0),self.memory_linv[index](memory_1d))
            lambda_bias_linear = torch.matmul(self.linbk[index](memory_1d).transpose(1,0), self.memory_linv[index](memory_1d))
            memory_bias_matrix_linear.append(lambda_bias_linear)
            memory_global_matrix_linear.append(lambda_layer_linear)

        global_memory_lambda_matrix_linear = torch.stack(memory_global_matrix_linear)
        global_memory_lambda_matrix_bias_linear = torch.stack(memory_bias_matrix_linear)
        

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        output_context = self.memory_linq[index](hidden_states) * global_memory_lambda_matrix_linear + global_memory_lambda_matrix_bias_linear
        
        return output_context, hidden_size





class BertHistoryGenerator7(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.m_ = nn.Softmax(dim=0)
        self.act_fn = ACT2FN['gelu']
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(13)])

        self.memory_linv = nn.ModuleList([nn.Linear(config.hidden_size, 128) for _ in range(13)])
        self.memory_link = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.linbk = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])
        
        self.memory_linv2 = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(13)])
        self.memory_link2 = nn.ModuleList([nn.Linear(config.hidden_size, 128) for _ in range(13)])
        self.linbk2 = nn.ModuleList([nn.Linear(config.hidden_size,1) for _ in range(13)])

        self.m2 = nn.Softmax(dim=1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, input_att_mask ,end=False):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, input_att_mask, end)
        
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)


        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index, input_att_mask=None, end=False):  

        memory_global_matrix_linear = []
        memory_bias_matrix_linear = []  

        memory_global_matrix_up = []
        memory_bias_matrix_up = []

        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 32 else 32, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]

            lambda_layer_linear=torch.matmul(self.m_(self.memory_link[index](memory_1d)).transpose(1,0),self.memory_linv[index](memory_1d))
            lambda_bias_linear = torch.matmul(self.m_(self.linbk[index](memory_1d)).transpose(1,0), self.memory_linv[index](memory_1d))
            
            lambda_layer_linear_up=torch.matmul(self.m_(self.memory_link2[index](memory_1d)).transpose(1,0),self.memory_linv2[index](memory_1d))
            lambda_bias_linear_up = torch.matmul(self.m_(self.linbk2[index](memory_1d)).transpose(1,0), self.memory_linv2[index](memory_1d))
            
            memory_bias_matrix_linear.append(lambda_bias_linear)
            memory_global_matrix_linear.append(lambda_layer_linear)

            memory_bias_matrix_up.append(lambda_bias_linear_up)
            memory_global_matrix_up.append(lambda_layer_linear_up)

        global_memory_lambda_matrix_linear = torch.stack(memory_global_matrix_linear)
        global_memory_lambda_matrix_bias_linear = torch.stack(memory_bias_matrix_linear)

        global_memory_lambda_matrix_linear2 = torch.stack(memory_global_matrix_up)
        global_memory_lambda_matrix_bias_linear2 = torch.stack(memory_bias_matrix_up)

        

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        output_context = torch.bmm(hidden_states, global_memory_lambda_matrix_linear) + global_memory_lambda_matrix_bias_linear
        # hidden = self.act_fn(output_context)
        # output_context = torch.bmm(hidden, global_memory_lambda_matrix_linear2) + global_memory_lambda_matrix_bias_linear2


        return output_context, hidden_size




class BertHistoryGenerator8(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.m_ = nn.Softmax(dim=0)
        self.act_fn = ACT2FN['gelu']
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.memory_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.memory_linq = nn.Linear(config.hidden_size, 128)
        self.memory_linv = nn.Linear(config.hidden_size, 128)
        self.memory_link = nn.Linear(config.hidden_size, 1)
        self.linbk = nn.Linear(config.hidden_size,1)
        
        self.memory_linq2 = nn.Linear(128, config.hidden_size)
        self.memory_linv2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_link2 = nn.Linear(config.hidden_size, 1)

        # self.memory_linv3 = nn.Linear(128, config.hidden_size)
        self.linbk2 = nn.Linear(config.hidden_size,1)

        self.m2 = nn.Softmax(dim=1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,hidden_states, memory, memory_len, input_len):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len)
        
        hidden_states_tmp2 = hidden_states_tmp1 + hidden_states


        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len):  

        memory_global_matrix_linear = []
        memory_bias_matrix_linear = []  

        memory_global_matrix_up = []
        memory_bias_matrix_up = []

        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 32 else 32, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]

            lambda_layer_linear=torch.matmul(self.m_(self.memory_link(memory_1d)).transpose(1,0),self.memory_linv(memory_1d))
            lambda_bias_linear = torch.matmul(self.m_(self.linbk(memory_1d)).transpose(1,0), self.memory_linv(memory_1d))
            
            lambda_layer_linear_up=torch.matmul(self.m_(self.memory_link2(memory_1d)).transpose(1,0),self.memory_linv2(memory_1d))
            lambda_bias_linear_up = torch.matmul(self.m_(self.linbk2(memory_1d)).transpose(1,0), self.memory_linv2(memory_1d))
            
            memory_bias_matrix_linear.append(lambda_bias_linear)
            memory_global_matrix_linear.append(lambda_layer_linear)

            memory_bias_matrix_up.append(lambda_bias_linear_up)
            memory_global_matrix_up.append(lambda_layer_linear_up)

        global_memory_lambda_matrix_linear = torch.stack(memory_global_matrix_linear)
        global_memory_lambda_matrix_bias_linear = torch.stack(memory_bias_matrix_linear)

        global_memory_lambda_matrix_linear2 = torch.stack(memory_global_matrix_up)
        global_memory_lambda_matrix_bias_linear2 = torch.stack(memory_bias_matrix_up)

        

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]
        # output_context = self.memory_linv(hidden_states)

        output_context = self.memory_linq(hidden_states) * global_memory_lambda_matrix_linear + global_memory_lambda_matrix_bias_linear
        # hidden = self.act_fn(output_context)
        output_context = self.memory_linq2(output_context) * global_memory_lambda_matrix_linear2 + global_memory_lambda_matrix_bias_linear2
        # output_context = self.memory_linv3(hidden)

        return output_context, hidden_size



class BertHistoryGenerator6(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.m_ = nn.Softmax(dim=0)
        self.act_fn = ACT2FN['gelu']
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.memory_LayerNorm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(13)])

        self.memory = torch.nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.linear3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.m2 = nn.Softmax(dim=1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index, input_att_mask):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, input_att_mask)
        hidden_states_tmp2 = self.memory_LayerNorm[index](hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index, input_att_mask=None):  

        memory_global_matrix_linear = []
        memory_bias_matrix_linear = []  

        for i in range(len(hidden_states)):
            memory_1d = memory[i][:history_len[i]]
            input_1d = hidden_states[i][:input_len[i]]

            masking = torch.matmul(memory_1d, input_1d.transpose(-1, -2))
            max_masking = torch.max(masking,dim=1)[0].squeeze(-1)
            selected_value, selected_index = torch.topk(max_masking,k=len(max_masking) if len(max_masking) < 32 else 32, sorted=True)
            selected_index = torch.sort(selected_index,dim=0)[0]
            memory_1d = memory_1d[selected_index]

            output, h_n = self.memory(memory_1d[None,:])
            lambda_layer_linear = self.linear(output[0,-1][None, :])
            lambda_bias_linear = self.linear2(output[0,-1][None, :])

            memory_bias_matrix_linear.append(lambda_bias_linear)
            memory_global_matrix_linear.append(lambda_layer_linear)

        global_memory_lambda_matrix_linear = torch.stack(memory_global_matrix_linear)
        global_memory_lambda_matrix_bias_linear = torch.stack(memory_bias_matrix_linear)
        

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] 
        hidden_size  = hidden_states.shape[2]

        output_context = self.act_fn(self.linear3(hidden_states)) * global_memory_lambda_matrix_linear + global_memory_lambda_matrix_bias_linear
        
        return output_context, hidden_size



class BertInjectMemoryLG(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        self.memory_query= nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size) for _ in range(1)])
        self.memory_value = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_key = nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size)

        self.memory_left_query = nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size)
        self.memory_left_key = nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size)
        self.memory_right_query = nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size)
        self.memory_right_key = nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size)
        self.memory_local_key = nn.Linear(config.hidden_size, config.hidden_size//config.bottleneck_size) 
        self.memory_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.g = nn.Parameter(torch.randn(config.hidden_size//config.bottleneck_size,1))
        self.h = nn.Parameter(torch.randn(config.hidden_size//config.bottleneck_size,1))
        self.activation=ACT2FN['gelu']
        
        self.memory_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.m_ = nn.Softmax(dim=0)
        self.m2 = nn.Softmax(dim=1)
        self.m3 = nn.Softmax(dim=2)
    
    def forward(self,hidden_states, memory, memory_len, input_len, index):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, query=self.memory_query,key=self.memory_key,value=self.memory_value)
        hidden_states_tmp2 = self.memory_LayerNorm(hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index,query=None, key=None, value=None):
        

        # memory_global_matrix = []
        memory_GL_matrix = []
        for i in range(len(hidden_states)):
            if value is not None and key is not None:

                memory_1d_all = memory[i][:history_len[i]]
                ValueVector=value(memory_1d_all)

                #global probablity distribution
                global_lambda = key(memory_1d_all).transpose(1,0).unsqueeze(0)

                # modeling local information
                outputL= torch.matmul(self.memory_left_query(hidden_states[i]), self.memory_left_key(memory_1d_all).transpose(1,0))
                outputL /= torch.sqrt(torch.FloatTensor([outputL.shape[-1]]).to(memory_1d_all.device))
                outputL_distribution = self.m2(outputL)
                
                outputR = torch.matmul(self.memory_right_query(hidden_states[i]), self.memory_right_key(memory_1d_all).transpose(1,0))
                outputR /= torch.sqrt(torch.FloatTensor([outputR.shape[-1]]).to(memory_1d_all.device))
                outputR_distribution = self.m2(outputR)

                L = torch.ones((memory_1d_all.shape[0],memory_1d_all.shape[0])).to(memory_1d_all.device)
                L = torch.triu(L, diagonal=0)
                mask = torch.matmul(outputL_distribution, L) * torch.matmul(outputR_distribution, L.transpose(0,1)) + torch.matmul(outputL_distribution,L.transpose(0,1)) * torch.matmul(outputR_distribution, L)
                local_lambda = (self.memory_local_key(memory_1d_all)[None, :] * mask[:,:,None]).transpose(1,2)

                logits = (global_lambda + local_lambda) / torch.sqrt(torch.FloatTensor([outputR.shape[-1]]).to(memory_1d_all.device))
                GL_lambda=self.m3(logits)
                lambda_all_matrix=torch.matmul(GL_lambda, ValueVector)
                lambda_all_matrix=self.memory_project(self.activation(lambda_all_matrix))
                memory_GL_matrix.append(lambda_all_matrix)

            elif key is not None:
                lambda_layer=torch.matmul(key(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])
            
            elif value is not None:
                lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),value(memory[i][:history_len[i]]))
            
            else:
                lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])

        local_memory_lambda_matrix = torch.stack(memory_GL_matrix)

        if query is not None:
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1] 
            hidden_size  = hidden_states.shape[2]
            # global information
            
            # local information
            local_memory_lambda_matrix_b = torch._weight_norm(local_memory_lambda_matrix,self.h, -1)
            

            # gate  = self.m2(self.gate_param(hidden_states))
            query_vector = query[index](hidden_states)
            hidden_states_tmp = self.activation(query_vector)

            # output1 = torch.bmm(hidden_states_tmp, global_memory_lambda_matrix_a)
            
            output_tensor = []
            for i in range(len(hidden_states_tmp)):
                output=torch.bmm(hidden_states_tmp[i][:,None,:],  local_memory_lambda_matrix_b[i]).squeeze(1)
                output_tensor.append(output)
            output2 = torch.stack(output_tensor)
            
            output =output2 #+output1  
            
            return output, local_memory_lambda_matrix_b
        else:
            global_memory_lambda_matrix= None
            output = torch.bmm( hidden_states * self.m2(query[index](hidden_states)), global_memory_lambda_matrix)
            return output, global_memory_lambda_matrix

class BertInjectMemoryAttention(nn.Module):
    def __init__(self, config, value_size):
        super().__init__()
        self.config = config
        
    
    def forward(self,hidden_states, memory, memory_len, input_len, index):

        hidden_states_tmp1, lambda_matrix = self.memory_operator(hidden_states, memory, memory_len, input_len, index, query=self.memory_query,key=self.memory_key,value=self.memory_value)
        hidden_states_tmp2 = self.memory_LayerNorm(hidden_states_tmp1 + hidden_states)

        return hidden_states_tmp2, lambda_matrix


    def memory_operator(self, hidden_states, memory, history_len, input_len, index,query=None, key=None, value=None):
        

        memory_global_matrix = []
        memory_local_matrix = []
        for i in range(len(hidden_states)):
            if value is not None and key is not None:
                memory_1d = memory[i][:history_len[i]]
                lambda_layer=torch.matmul(self.m_(key(memory_1d).transpose(1,0)),value(memory_1d))
                lambda_layer=self.memory_project(self.activation(lambda_layer))
            elif key is not None:
                lambda_layer=torch.matmul(key(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])
            
            elif value is not None:
                lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),value(memory[i][:history_len[i]]))
            
            else:
                lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])

            memory_global_matrix.append(lambda_layer)
        global_memory_lambda_matrix = torch.stack(memory_global_matrix)

        if query is not None:
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1] 
            hidden_size  = hidden_states.shape[2]

            global_memory_lambda_matrix_a = torch._weight_norm(global_memory_lambda_matrix, self.g,-1)
            query_vector = query[index](hidden_states)
            hidden_states = self.activation(query_vector)
            output2 = torch.bmm(hidden_states, global_memory_lambda_matrix_a)
            
            output = output2
            
            return output, global_memory_lambda_matrix
        else:
            output = torch.bmm( hidden_states * self.m2(query[index](hidden_states)), global_memory_lambda_matrix)
            return output, global_memory_lambda_matrix

class HisBERTModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderHisBERT(config)
        self.memory_module=BertHistoryGenerator3(config, config.hidden_size)
        self.pooler = BertPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)

        encoder_outputs_mem = self.encoder(
            memory_embed,
            self.embeddings.position_embeddings,
            embedding_output_tmp,
            memory_module=self.memory_module,
            memory_len=input_len,
            input_len=memory_len,
            memory_att_mask=extended_attention_mask,
            attention_mask=extended_memory_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            index=1,
        )

        encoder_outputs = self.encoder(
            embedding_output_tmp,
            self.embeddings.position_embeddings,
            memory_embed,
            memory_module=self.memory_module,
            memory_len=memory_len,
            input_len=input_len,
            memory_att_mask=extended_memory_attention_mask,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=memory_embed,
            encoder_attention_mask=extended_memory_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            index=1,
        )


        sequence_output_mem = encoder_outputs_mem[0]
        sequence_output = encoder_outputs[0]

        encoder_outputs_mem = self.encoder(
            sequence_output_mem,
            self.embeddings.position_embeddings,
            sequence_output,
            memory_module=self.memory_module,
            memory_len=input_len,
            input_len=memory_len,
            memory_att_mask=extended_attention_mask,
            attention_mask=extended_memory_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=sequence_output,
            encoder_attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start=1,
            index=11,
        )

        sequence_output_mem_11 = encoder_outputs_mem[0]

        
        encoder_outputs = self.encoder(
            sequence_output,
            self.embeddings.position_embeddings,
            sequence_output_mem,
            memory_module=self.memory_module,
            memory_len=memory_len,
            input_len=input_len,
            memory_att_mask=extended_memory_attention_mask,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=sequence_output_mem,
            encoder_attention_mask=extended_memory_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            index=11,
            start=1,
        )

        sequence_output = encoder_outputs[0]

        encoder_outputs = self.encoder(
            sequence_output,
            self.embeddings.position_embeddings,
            sequence_output_mem_11,
            memory_module=self.memory_module,
            memory_len=memory_len,
            input_len=input_len,
            memory_att_mask=extended_memory_attention_mask,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=sequence_output_mem_11,
            encoder_attention_mask=extended_memory_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start=11,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class BertModelMemory2(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module =BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embedding_output, memory_lambda = self.memory_module(memory_embed, memory_embed_query, memory_len_query, memory_len,0)
        embedding_output, memory_lambda = self.memory_module(embedding_output_tmp, memory_embed, memory_len, input_len,0)


        encoder_outputs_mem = self.encoder(
            memory_embedding_output,
            self.embeddings.position_embeddings,
            embedding_output_tmp,
            memory_module=self.memory_module,
            memory_len=input_len,
            input_len=memory_len,
            memory_att_mask=extended_attention_mask,
            attention_mask=extended_memory_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

        )

        encoder_outputs = self.encoder(
            embedding_output,
            self.embeddings.position_embeddings,
            memory_embed,
            memory_module=self.memory_module,
            memory_len=memory_len,
            input_len=input_len,
            memory_att_mask=extended_memory_attention_mask,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=memory_embed,
            encoder_attention_mask=extended_memory_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output_mem = encoder_outputs_mem[0]
        sequence_output, _ =self.memory_module(sequence_output, sequence_output_mem, memory_len, input_len, 0)
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertModelMemory3(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            encoder_outputs = self.encoder(
                embedding_output,
                self.embeddings.position_embeddings,
                memory_embed,
                memory_module=self.memory_module,
                memory_len=memory_len,
                input_len=input_len,
                memory_att_mask=extended_memory_attention_mask,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=memory_embed,
                encoder_attention_mask=extended_memory_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
            )

            embedding_output = encoder_outputs[0]
            memory_embed = encoder_outputs_mem[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )





class BertModelMemory10(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i==1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed = encoder_outputs_mem[0]
            embedding_output = encoder_outputs[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertModelMemory11(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i==0:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed = encoder_outputs_mem[0]
            embedding_output = encoder_outputs[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class BertModelMemory13(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.embeddings_mem = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.encoder_generator = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings_mem(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings_mem(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder_generator(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i==0:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed = encoder_outputs_mem[0]
            embedding_output = encoder_outputs[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )






class BertModelMemory14(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator10(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i==0:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
            else:
                embedding_output = embedding_output_tmp
                memory_embed = memory_embed_tmp

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertModelMemory15(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)

        encoder_outputs_mem_query = self.encoder(
                memory_embed_query,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len_query,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_query_att_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=0,
                index=1,
                )

        memory_embed = memory_embed_tmp
        memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            if i == 11 or i == 0:
                if i == 0:
                    memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                if i == 11:
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                    memory_embed = memory_embed_tmp
            else:
                memory_embed = memory_embed_tmp
                embedding_output = embedding_output_tmp
        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class BertModelMemory22(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemoryMarker(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        history_ids=None,
        history_doc_ids=None,
        history_mem_ids=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=history_ids)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=history_mem_ids)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=history_doc_ids)

        encoder_outputs_mem_query = self.encoder(
            memory_embed_query,
            self.embeddings.position_embeddings,
            embedding_output,
            memory_module=self.memory_module,
            memory_len=input_len,
            input_len=memory_len_query,
            memory_att_mask=extended_attention_mask,
            attention_mask=extended_query_att_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start=0,
            index=1,
            )

        memory_embed = memory_embed_tmp
        memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            if i == 11 or i == 0:
                if i == 0:
                    memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                if i == 11:
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                    memory_embed = memory_embed_tmp
            else:
                memory_embed = memory_embed_tmp
                embedding_output = embedding_output_tmp
        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertModelMemory23(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.embeddings_mem = BertEmbeddingsMemory2(config)
        self.encoder_mem = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator22(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        # with torch.no_grad():
        memory_embed_tmp = self.embeddings_mem(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed = memory_embed_tmp

        for i in range(self.config.num_hidden_layers):
            
            # with torch.no_grad():
            encoder_outputs_mem = self.encoder_mem(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 


            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]

            memory_embed = memory_embed_tmp
            embedding_output = embedding_output_tmp

        pooler_output = self.pooler(embedding_output)
        embedding_output, output = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)
        sequence_output = embedding_output

        if not return_dict:
            return (sequence_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



class BertModelMemory21(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)

        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)

        encoder_outputs_mem_query = self.encoder(
                memory_embed_query,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len_query,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_query_att_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=0,
                index=1,
                )

        memory_embed = memory_embed_tmp
        memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 

            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            if i == 11 or i == 0:
                if i == 0:
                    memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                if i == 11:
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                    memory_embed = memory_embed_tmp
            else:
                memory_embed = memory_embed_tmp
                embedding_output = embedding_output_tmp
        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )










class BertModelMemory20(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)

        encoder_outputs_mem_query = self.encoder(
                memory_embed_query,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len_query,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_query_att_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=0,
                index=1,
                )

        memory_embed = memory_embed_tmp
        memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            if i == 11 or i == 0:
                if i == 0:
                    memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                if i == 11:
                    embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                    memory_embed = memory_embed_tmp
            else:
                memory_embed = memory_embed_tmp
                embedding_output = embedding_output_tmp
        
        encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
        encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

        encoder_outputs = self.encoder(
            embedding_output,
            self.embeddings.position_embeddings,
            memory_embed,
            memory_module=self.memory_module,
            memory_len=memory_len,
            input_len=input_len,
            memory_att_mask=extended_memory_attention_mask,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_memory_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start=i,
            index=i+1,
            cross_att=True,
        )

        embedding_output_tmp = encoder_outputs[0]
        sequence_output = embedding_output_tmp
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertModelMemory16(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        # encoder_outputs_mem_query = self.encoder(
        #         memory_embed_query,
        #         self.embeddings.position_embeddings,
        #         embedding_output,
        #         memory_module=self.memory_module,
        #         memory_len=input_len,
        #         input_len=memory_len_query,
        #         memory_att_mask=extended_attention_mask,
        #         attention_mask=extended_query_att_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=None,
        #         encoder_attention_mask=None,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #         start=0,
        #         index=1,
        #         )

        # memory_embed = memory_embed_tmp
        # memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, 0, extended_attention_mask)
                memory_embed = memory_embed_tmp
            else:
                memory_embed = memory_embed_tmp
                embedding_output = embedding_output_tmp
        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )









class BertModelMemory17(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        # memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        # memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        # embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        # encoder_outputs_mem_query = self.encoder(
        #         memory_embed_query,
        #         self.embeddings.position_embeddings,
        #         embedding_output,
        #         memory_module=self.memory_module,
        #         memory_len=input_len,
        #         input_len=memory_len_query,
        #         memory_att_mask=extended_attention_mask,
        #         attention_mask=extended_query_att_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=None,
        #         encoder_attention_mask=None,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #         start=0,
        #         index=1,
        #         )

        memory_embed = memory_embed_tmp
        embedding_output = embedding_output_tmp
        # memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            memory_embed = memory_embed_tmp
            embedding_output = embedding_output_tmp

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )





class BertModelMemory18(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        # memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        # memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        # embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        # encoder_outputs_mem_query = self.encoder(
        #         memory_embed_query,
        #         self.embeddings.position_embeddings,
        #         embedding_output,
        #         memory_module=self.memory_module,
        #         memory_len=input_len,
        #         input_len=memory_len_query,
        #         memory_att_mask=extended_attention_mask,
        #         attention_mask=extended_query_att_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=None,
        #         encoder_attention_mask=None,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #         start=0,
        #         index=1,
        #         )

        memory_embed = memory_embed_tmp
        embedding_output = embedding_output_tmp
        # memory_embed_query = encoder_outputs_mem_query[0]


        for i in range(self.config.num_hidden_layers):
            
            if i == 1:
                encoder_hidden_states_mem = torch.cat((memory_embed, embedding_output), dim=1)
                encoder_memory_attention_mask_mem = torch.cat((extended_memory_attention_mask, extended_attention_mask), dim=-1)                
                
                encoder_outputs_mem = self.encoder(
                    memory_embed,
                    self.embeddings.position_embeddings,
                    embedding_output,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states_mem,
                    encoder_attention_mask=encoder_memory_attention_mask_mem,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs_mem = self.encoder(
                    memory_embed,
                    self.embeddings.position_embeddings,
                    embedding_output,
                    memory_module=self.memory_module,
                    memory_len=input_len,
                    input_len=memory_len,
                    memory_att_mask=extended_attention_mask,
                    attention_mask=extended_memory_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    )

            if i == 11 or i== 1:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed_tmp = encoder_outputs_mem[0]
            embedding_output_tmp = encoder_outputs[0]
            memory_embed = memory_embed_tmp
            embedding_output = embedding_output_tmp

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class BertModelMemory115(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )


            encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed = encoder_outputs_mem[0]
            embedding_output = encoder_outputs[0]
            if i == 11:
                embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class BertModelMemory12(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.encoder = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator9(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
    
            encoder_outputs_mem = self.encoder(
                memory_embed,
                self.embeddings.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            if i == 11 or i==0:
                encoder_hidden_states = torch.cat((embedding_output, memory_embed), dim=1)
                encoder_memory_attention_mask = torch.cat((extended_attention_mask, extended_memory_attention_mask), dim=-1)

                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                    cross_att=True,
                )

            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    self.embeddings.position_embeddings,
                    memory_embed,
                    memory_module=self.memory_module,
                    memory_len=memory_len,
                    input_len=input_len,
                    memory_att_mask=extended_memory_attention_mask,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=memory_embed,
                    encoder_attention_mask=extended_memory_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                ) 
            memory_embed = encoder_outputs_mem[0]
            embedding_output = encoder_outputs[0]
            if i == 10:
                embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, 0, extended_attention_mask)

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )





class BertModelMemory4(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.embeddings_mem = BertEmbeddingsMemory2(config)

        self.encoder = BertEncoderMemory(config)
        self.encoder_generator = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator3(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)

        for i in range(self.config.num_hidden_layers):
            
            if i+1 < self.config.num_hidden_layers:
                encoder_outputs_mem = self.encoder_generator(
                    memory_embed,
                    self.embeddings_mem.position_embeddings,
                    embedding_output_tmp,
                    memory_module=self.memory_module,
                    memory_len=input_len,
                    input_len=memory_len,
                    memory_att_mask=extended_attention_mask,
                    attention_mask=extended_memory_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    start=i,
                    index=i+1,
                )

            encoder_outputs = self.encoder(
                embedding_output,
                self.embeddings.position_embeddings,
                memory_embed,
                memory_module=self.memory_module,
                memory_len=memory_len,
                input_len=input_len,
                memory_att_mask=extended_memory_attention_mask,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=memory_embed,
                encoder_attention_mask=extended_memory_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
            )

            embedding_output = encoder_outputs[0]
            memory_embed = encoder_outputs_mem[0]
            if i+1 < self.config.num_hidden_layers:
                if i == 0 or i == 11:
                    embedding_output, _ = self.memory_module(embedding_output, memory_embed, memory_len, input_len, i+1, extended_attention_mask)
            

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )




class BertModelMemory5(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsMemory2(config)
        self.embeddings_mem = BertEmbeddingsMemory2(config)

        self.encoder = BertEncoderMemory(config)
        self.encoder_generator = BertEncoderMemory(config)
        self.pooler = BertPooler(config)
        self.memory_module = BertHistoryGenerator7(config, config.hidden_size)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        memory_input_ids0=None,
        memory_input_ids1=None,
        input_len=None,
        memory_len=None,
        memory_attention_mask=None,
        memory=None,
        memory_segment=None,
        memory_position_id=None,
        memory_query=None,
        memory_len_query=None,
        memory_segment_query=None,
        input_ids=None,
        query_att_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,

    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_query_att_mask = self.get_extended_attention_mask(query_att_mask, memory_query.size(), device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        extended_memory_attention_mask: torch.Tensor = self.get_extended_attention_mask(memory_attention_mask, memory.size(), device)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output_tmp = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        ,memory_ids=memory_input_ids0,memory=False)
        memory_embed_tmp = self.embeddings_mem(input_ids=memory, token_type_ids=memory_segment, memory_ids=memory_input_ids1,memory=False)
        memory_embed_query = self.embeddings_mem(input_ids=memory_query, token_type_ids=memory_segment_query, memory_ids=None,memory=False)
        memory_embed, _ = self.memory_module(memory_embed_tmp, memory_embed_query, memory_len_query, memory_len,0, extended_memory_attention_mask)
        embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len,0, extended_attention_mask)



        for i in range(self.config.num_hidden_layers):
            
            encoder_outputs_mem = self.encoder_generator(
                memory_embed,
                self.embeddings_mem.position_embeddings,
                embedding_output_tmp,
                memory_module=self.memory_module,
                memory_len=input_len,
                input_len=memory_len,
                memory_att_mask=extended_attention_mask,
                attention_mask=extended_memory_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
                )

            encoder_outputs = self.encoder(
                embedding_output,
                self.embeddings.position_embeddings,
                memory_embed,
                memory_module=self.memory_module,
                memory_len=memory_len,
                input_len=input_len,
                memory_att_mask=extended_memory_attention_mask,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=memory_embed,
                encoder_attention_mask=extended_memory_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start=i,
                index=i+1,
            )

            embedding_output_tmp = encoder_outputs[0]
            memory_embed_tmp = encoder_outputs_mem[0]
            
            if i < 11:
                embedding_output, _ = self.memory_module(embedding_output_tmp, memory_embed_tmp, memory_len, input_len, i+1, extended_attention_mask)
                memory_embed = memory_embed_tmp
            else:
                memory_embed = memory_embed_tmp
                embedding_output = embedding_output_tmp

        sequence_output = embedding_output
        pooled_output = self.pooler(sequence_output)

    
        if not return_dict:
            return (sequence_output, pooled_output, ) + encoder_outputs[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning. """, BERT_START_DOCSTRING
)
class BertLMHeadModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **maked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            n ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> config.is_decoder = True
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config, return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            number = 0
            lm_loss = torch.tensor(0.).to(labels.device)
            for i in range(len(labels)):
                if labels[i][0] == 101 and labels[i][1] == 102:
                    continue
                number +=1
                shifted_prediction_scores = prediction_scores[i, :-1, :].contiguous()
                label_view = labels[i, 1:].contiguous()
                loss_fct = CrossEntropyLoss(ignore_index=0,reduction="sum")
                lm_loss += loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), label_view.view(-1))
            if number >0:
                lm_loss /= number
            # import ipdb; ipdb.set_trace()

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """,
    BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring).  Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', return_dict=True)

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, next_sentence_label=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), next_sentence_label.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
