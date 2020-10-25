import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from argparse import Namespace
from torch.nn import Linear, Dropout
import math
import torch.nn.functional as F
import copy

import IPython
import pdb

LayerNorm = torch.nn.LayerNorm

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu





def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def apply_chunking_to_forward(
forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
):
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked.
    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`foward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape
    assert all(
        input_tensor.shape == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
}

def prune_linear_layer(layer: torch.nn.Linear, index: torch.LongTensor, dim: int = 0):
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
):
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index




class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers,config):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

    def forward(self, hidden_state, att_mask, relative_embed):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = hidden_state

        for mod in self.layers:
            output = mod(output, att_mask, relative_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, config, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        # Implementation of Feedforward model
        self.linear1 = Linear(config.hidden_size, config.intermediate_size)
        self.dropout = Dropout(config.dropout)
        self.linear2 = Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = Dropout(config.dropout)
        self.dropout2 = Dropout(config.dropout)

        self.activation = ACT2FN[activation]

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, hidden_states, attention_mask, relative_embed):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 =  self.self_attn(
            hidden_states,
            relative_embed,
            attention_mask=attention_mask,
        )
        src = hidden_states + self.dropout1(src2[0])
        src = self.LayerNorm1(hidden_states)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2[0])
        src = self.LayerNorm2(src)
        return src

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

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)
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
        relative_embed,
        attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            relative_embed,
            attention_mask=attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class SelfAttention(nn.Module):
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
                self.pos_proj = Linear(config.hidden_size, self.all_head_size, bias=False)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = Linear(config.hidden_size, self.all_head_size)
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embeddings,
        relative_pos=None,
        attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        
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
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        if rel_embeddings is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)            


        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)



        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
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
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
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

class SelfOutput(nn.Module):
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


class CoherenceEncoder(nn.Module):
    def __init__(self,config):
        super(CoherenceEncoder, self).__init__()
        layer = TransformerEncoderLayer(config,activation="gelu")
        self.model = TransformerEncoder(layer,config.n_layer,config)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self,token_embed, seg_embed, attention_mask, relative_embed):
        embedding = token_embed + seg_embed
        embedding = self.LayerNorm(embedding)
        out = self.model(embedding, attention_mask, relative_embed)

        return out


class PassageEncoder(nn.Module):
    def __init__(self, config):
        super(PassageEncoder, self).__init__()
        layer = TransformerEncoderLayer(config,activation="gelu")
        self.model = TransformerEncoder(layer,config.n_layer,config)

    def forward(self, token_embed, attention_mask, relative_embed):
         embedding = token_embed
         out = self.model(embedding, attention_mask, relative_embed)

         return out


class ReasonNet(nn.Module):
    def __init__(self, config):
        super(ReasonNet, self).__init__()
        layer = TransformerEncoderLayer(config,activation="gelu")
        self.model = TransformerEncoder(layer,config.n_layer,config)

    def forward(self, hidden_state, attention_mask, relative_embed):
        
        output = self.model(hidden_state, attention_mask, relative_embed)
    
        return output


class CQAModel(nn.Module):
    def __init__(self, config):
        super(CQAModel, self).__init__()

        #q_layer, c_layer, r_layer, n_dim, nhead
        config = Namespace(**config)
    
        self.pretrained_model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding = self.pretrained_model.embeddings.word_embeddings
        self.relative_seg_embed = nn.Embedding(4, config.hidden_size) # [ -2, -1, 0 , 1] -> [ 0, 1, 2, 3]
        self.relative_pos_embed = nn.Embedding(config.max_relative_positions*2,config.hidden_size)
        self.qp_seg_embed = nn.Embedding(3, config.hidden_size) # [q,a, pad]
        self.q_encoder = CoherenceEncoder(Namespace(**config.q_config))
        self.p_encoder = PassageEncoder(Namespace(**config.p_config))
        self.interNet  = ReasonNet(Namespace(**config.r_config))
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.adam_epsilon)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act, start_positions, end_positions):
        """
        q_ids / q_segs / q_att_masks: 
            size: B x N x 1: integer matrix
            N will be the max length of question in batch  

        c_ids / c_att_masks:
            size : B x M x 1: integer matrix
            M will be the max length of context in batch 
        """ 


        q_embed = self.embedding(q_ids)
        c_embed = self.embedding(c_ids)

        seg_embed = self.relative_seg_embed(q_segs+2)

        device = q_embed.device

        q_att_masks = self.get_extended_attention_mask(q_att_masks, q_embed.size()[:-1], device)
        c_att_masks = self.get_extended_attention_mask(c_att_masks, c_embed.size()[:-1], device)


        q_reps = self.q_encoder(q_embed, seg_embed,q_att_masks,self.relative_pos_embed.weight)

        c_reps = self.p_encoder(c_embed, c_att_masks,self.relative_pos_embed.weight)


        
        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_reps)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]
            q_rep = q_reps[i][q_start[i]:q_len[i]]
            c_rep = c_reps[i][:c_len[i]]
            
            q_seg = torch.zeros_like(q_item).fill_(2)
            c_seg = torch.zeros_like(c_item).fill_(1)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_rep,q_rep],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_sep = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        seg_embedding = self.qp_seg_embed(reason_seg)
        reason_input = reason_sep + seg_embedding
        reason_input = self.LayerNorm(reason_input)

        reason_mask = self.get_extended_attention_mask(reason_mask, reason_input.size()[:-1], device)

        final_output = self.interNet(reason_input, reason_mask, self.relative_pos_embed.weight)

        final_out = []
        for i in range(len(final_output)):
            final_out.append(final_output[i][:c_len[i]])
            final_rep = pad_sequence(final_out, batch_first=True)
        
        logits=self.qa_output(final_rep)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # IPython.embed()
        # pdb.set_trace()

        return  start_logits, end_logits

    def get_extended_attention_mask(self,attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask





        

        


        
        



        