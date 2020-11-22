import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, AlbertModelRelative, RobertaModelRelative, BertConfig, BertModelRelative, AlbertConfig, RobertaConfig, BertModelinitial_from_mu_sigma, BertModelMemory
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


class BERTQA_initial(nn.Module):
    def __init__(self, config):
        super(BERTQA_initial, self).__init__()

        config = Namespace(**config)
        self.model_config = BertConfig.from_pretrained("bert-base-uncased")
        self.model_config.update(config.model_config)
        self.pretrained_model = BertModelinitial_from_mu_sigma.from_pretrained("bert-base-uncased", config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][1:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            sep_symbol = torch.nonzero((q_item==102),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output


class BERTQA_no_cross_att(nn.Module):
    def __init__(self, config):
        super(BERTQA_no_cross_att, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =BertConfig.from_pretrained("bert-base-uncased")
        self.model_config.update(config.model_config)
        self.pretrained_model = BertModelRelative.from_pretrained("bert-base-uncased", config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][1:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            sep_symbol = torch.nonzero((q_item==102),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output

class BERTQA_memory(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =BertConfig.from_pretrained("bert-base-uncased")
        self.model_config.update(config.model_config)
        self.pretrained_model = BertModelMemory.from_pretrained("bert-base-uncased", config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()
        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []
        memory_input = []
        mask_memory_att_input = []
        input_len = []
        seg_memory_input = []
        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            sep_symbol = torch.nonzero((q_item==102),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

            memory = q_ids[i][:q_start[i]]
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            mask_memory_att_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_init_seg = torch.ones_like(memory)
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        memory_ids = pad_sequence(memory_input,batch_first=True)
        input_total_length = torch.stack(input_len).squeeze(1)
        memory_att_mask = torch.stack(mask_memory_att_input).squeeze(1)
        memory_segment = pad_sequence(seg_memory_input, batch_first=True)


        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
            "input_len":input_total_length,
            "memory": memory_ids,
            "memory_attention_mask":memory_att_mask,
            "memory_segment": memory_segment
        }

        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output(hidden_states)

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output
    
    def memory_operator(self, hidden_states, memory, history_len):
        memory_matrix = []
        for i in range(len(hidden_states)):
            lambda_layer=torch.matmul(self.m_(memory[i][:history_len[i]]).transpose(1,0),memory[i][:history_len[i]])
            memory_matrix.append(lambda_layer)
        lambda_matrix=torch.stack(memory_matrix)
        output = torch.bmm(hidden_states, lambda_matrix)
        return output
    
class BERTQA_cross_att(nn.Module):
    def __init__(self, config):
        super(BERTQA_cross_att, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =BertConfig.from_pretrained("bert-base-uncased")
        self.model_config.update(config.model_config)
        self.pretrained_model = BertModelRelative.from_pretrained("bert-base-uncased", config=self.model_config)

        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][1:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            sep_symbol = torch.nonzero((q_item==102),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        reason_mask = reason_mask[:, None, :]
        reason_mask = reason_mask.repeat(1,reason_mask.shape[2],1)

        for i in range(len(q_ids)):
            # history just can see current question
            reason_mask[i][c_len[i]:c_len[i]+q_start[i]-1, :c_len[i]+q_start[i]-1] = 0
            # context can't see history turn
            reason_mask[i][:c_len[i], c_len[i]-1:c_len[i]+q_start[i]-1] = 0
            # current question can't see itself
            reason_mask[i][c_len[i]+q_start[i]-1:, c_len[i]+q_start[i]-1:] = 0
            reason_mask[i][0,1:c_len[i]] = 0



        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output


class BERTQA2_coqa(nn.Module):
    def __init__(self, config):
        super(BERTQA2_coqa, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.pretrained_model = BertModelRelative.from_pretrained("bert-base-uncased")
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            sep_symbol = torch.nonzero((q_item==102),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits, prediction)
        return ((total_loss,) + output) if total_loss is not None else output



class ALBERTQA_no_cross_att(nn.Module):
    def __init__(self, config):
        super(ALBERTQA_no_cross_att, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =AlbertConfig.from_pretrained("albert-base-v2")
        self.model_config.update(config.model_config)
        self.pretrained_model = AlbertModelRelative.from_pretrained("albert-base-v2",config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            cls_symbol = torch.nonzero((q_item==3),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)

            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)


        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output

class ALBERTQA_cross_att(nn.Module):
    def __init__(self, config):
        super(ALBERTQA_cross_att, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =AlbertConfig.from_pretrained("albert-base-v2")
        self.model_config.update(config.model_config)
        self.pretrained_model = AlbertModelRelative.from_pretrained("albert-base-v2",config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][1:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            cls_symbol = torch.nonzero((q_item==3),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(1)

            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        reason_mask = reason_mask[:, None, :]
        reason_mask = reason_mask.repeat(1,reason_mask.shape[2],1)

        for i in range(len(q_ids)):
            reason_mask[i][c_len[i]:c_len[i]+q_start[i], :c_len[i]+q_start[i]-1] = 0
            reason_mask[i][:c_len[i], c_len[i]:c_len[i]+q_start[i]-1] = 0
            reason_mask[i][c_len[i]+q_start[i]-1:, c_len[i]+q_start[i]-1:] = 0
            reason_mask[i][0,1:c_len[i]] = 0


        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output




class RobertaQA_no_cross_att(nn.Module):
    def __init__(self, config):
        super(RobertaQA_no_cross_att, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =AlbertConfig.from_pretrained("roberta-base")
        self.model_config.update(config.model_config)
        self.pretrained_model = RobertaModelRelative.from_pretrained("roberta-base", config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            cls_symbol = torch.nonzero((q_item==3),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(0)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output
    


class RobertaQA_cross_att(nn.Module):
    def __init__(self, config):
        super(RobertaQA_cross_att, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)
        self.model_config =AlbertConfig.from_pretrained("roberta-base")
        self.model_config.update(config.model_config)
        self.pretrained_model = RobertaModelRelative.from_pretrained("roberta-base", config=self.model_config)
        self.qa_output = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][1:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            cls_symbol = torch.nonzero((q_item==3),as_tuple=False)

            q_seg = torch.zeros_like(q_item).fill_(0)
            c_seg = torch.zeros_like(c_item).fill_(0)
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)

        reason_mask = reason_mask[:, None, :]
        reason_mask = reason_mask.repeat(1,reason_mask.shape[2],1)

        for i in range(len(q_ids)):
            reason_mask[i][c_len[i]:c_len[i]+q_start[i]-1, :c_len[i]+q_start[i]-1] = 0
            reason_mask[i][:c_len[i], c_len[i]:c_len[i]+q_start[i]-1] = 0
            reason_mask[i][c_len[i]+q_start[i]-1:, c_len[i]+q_start[i]-1:] = 0
            reason_mask[i][0,1:c_len[i]] = 0

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
        }

        last_reps = self.pretrained_model(**input_args)

        logits = self.qa_output(last_reps[0])

        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss ) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output