import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, AlbertModelRelative, \
    RobertaModelRelative, BertConfig, BertModelRelative,\
    AlbertConfig, RobertaConfig, BertModelinitial_from_mu_sigma, \
    BertModelMemory2, AlbertModelMemory, RobertaModelMemory, RobertaModelMemory2,BertInjectMemory

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

class BERTQA_memory_coqa(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory_coqa, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory2.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory2.from_pretrained(config.pretrained_name, config=self.model_config)
        
        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)
        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []
        memory_input = []
        memory_len_input = []
        input_len = []
        seg_memory_input = []
        memory_attention_mask_input = []
        memory_position_id = []
        memory_id0 = []
        memory_id1 = []
        memory_query = []
        memory_segment_query = []
        memory_len_query = []
        memory_attention_mask_query = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(torch.cat((cls_token,q_item),dim=0))
            memory_segment_query.append(torch.zeros_like(torch.cat((cls_token,q_item),dim=0)))
            memory_len_query.append(torch.LongTensor([len(q_item)+1]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(torch.cat((cls_token,q_item),dim=0)))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            
            if q_start[i] != 1:
                memory = q_ids[i][:q_start[i]]
            else:
                memory = q_ids[i][:q_start[i]] 
                sep = torch.LongTensor([102]).to(q_seg.device)
                memory = torch.cat((memory,sep),dim=0)
            memory_len_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_attention_mask_input.append(torch.ones_like(memory))

            # sep token id  == 102 
            sep_symbol = torch.nonzero((memory==102),as_tuple=False)
            
            if sep_symbol.shape[0] == 1:
                memory_init_seg = torch.zeros_like(memory)
            else:
                memory_init_seg_list = []
                start = 0
                for index in range(sep_symbol.squeeze(-1).shape[0]):
                    if index % 2 == 0:
                        memory_seg = torch.zeros_like(memory[start:sep_symbol[index]+1])
                    else:
                        memory_seg = torch.ones_like(memory[start:sep_symbol[index]+1])
                    start = sep_symbol[index]+1
                    memory_init_seg_list.append(memory_seg)
                memory_init_seg = torch.cat(memory_init_seg_list, dim=0)
            # memory_init_seg = torch.zeros_like(memory)
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)
            # memory_id1.append(torch.ones_like(memory))

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        memory_ids = pad_sequence(memory_input,batch_first=True)
        input_total_length = torch.stack(input_len).squeeze(1)
        memory_len = torch.stack(memory_len_input).squeeze(1)
        memory_segment = pad_sequence(seg_memory_input, batch_first=True)
        memory_attention_mask = pad_sequence(memory_attention_mask_input, batch_first=True)
        
        
        memory_segment_query = pad_sequence(memory_segment_query, batch_first=True)
        memory_length_query=torch.stack(memory_len_query)
        memory_query = pad_sequence(memory_query, batch_first=True)
        memory_attention_mask_query = pad_sequence(memory_attention_mask_query, batch_first=True)

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
            "input_len":input_total_length,
            "memory": memory_ids,
            "memory_len":memory_len,
            "memory_segment": memory_segment,
            "memory_attention_mask":memory_attention_mask,
            "memory_query": memory_query,
            "memory_len_query":memory_len_query,
            "memory_segment_query":memory_segment_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
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
            total_loss = (start_loss + end_loss + dialog_loss) /3

        output = (start_logits, end_logits, prediction)
        return ((total_loss,) + output) if total_loss is not None else output

class BERTQA_memory(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory2.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory2.from_pretrained(config.pretrained_name, config=self.model_config)


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []
        memory_input = []
        memory_len_input = []
        input_len = []
        seg_memory_input = []
        memory_attention_mask_input = []
        memory_position_id = []
        memory_query = []
        memory_segment_query = []
        memory_len_query = []
        memory_attention_mask_query = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(torch.cat((cls_token,q_item),dim=0))
            memory_segment_query.append(torch.zeros_like(torch.cat((cls_token,q_item),dim=0)))
            memory_len_query.append(torch.LongTensor([len(q_item)+1]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(torch.cat((cls_token,q_item),dim=0)))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            
            if q_start[i] != 1:
                memory = q_ids[i][:q_start[i]]
            else:
                memory = q_ids[i][:q_start[i]] 
                sep = torch.LongTensor([102]).to(q_seg.device)
                memory = torch.cat((memory,sep),dim=0)
            memory_len_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_attention_mask_input.append(torch.ones_like(memory))

            # sep token id  == 102 
            sep_symbol = torch.nonzero((memory==102),as_tuple=False)
            
            if sep_symbol.shape[0] == 1:
                memory_init_seg = torch.zeros_like(memory)
            else:
                memory_init_seg_list = []
                start = 0
                for index in range(sep_symbol.squeeze(-1).shape[0]):
                    if index % 2 == 0:
                        memory_seg = torch.zeros_like(memory[start:sep_symbol[index]+1])
                    else:
                        memory_seg = torch.ones_like(memory[start:sep_symbol[index]+1])
                    start = sep_symbol[index]+1
                    memory_init_seg_list.append(memory_seg)
                memory_init_seg = torch.cat(memory_init_seg_list, dim=0)
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)

        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        memory_ids = pad_sequence(memory_input,batch_first=True)
        input_total_length = torch.stack(input_len).squeeze(1)
        memory_len = torch.stack(memory_len_input).squeeze(1)
        memory_segment = pad_sequence(seg_memory_input, batch_first=True)
        memory_attention_mask = pad_sequence(memory_attention_mask_input, batch_first=True)
        
        
        memory_segment_query = pad_sequence(memory_segment_query, batch_first=True)
        memory_length_query=torch.stack(memory_len_query)
        memory_query = pad_sequence(memory_query, batch_first=True)
        memory_attention_mask_query = pad_sequence(memory_attention_mask_query, batch_first=True)

        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
            "input_len":input_total_length,
            "memory": memory_ids,
            "memory_len":memory_len,
            "memory_segment": memory_segment,
            "memory_attention_mask":memory_attention_mask,
            "memory_query": memory_query,
            "memory_len_query":memory_len_query,
            "memory_segment_query":memory_segment_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
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
            total_loss = (start_loss + end_loss + dialog_loss) /3

        output = (start_logits, end_logits, )
        return ((total_loss,) + output) if total_loss is not None else output
    

class ALBERTQA_memory(nn.Module):
    def __init__(self, config):
        super(ALBERTQA_memory, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =AlbertConfig.from_pretrained("albert-base-v2")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size})
            self.pretrained_model = AlbertModelMemory.from_pretrained("albert-base-v2",config=self.model_config)

        else:
            self.model_config =AlbertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size})
            self.pretrained_model = AlbertModelMemory.from_pretrained(config.pretrained_name,config=self.model_config)
        
        self.qa_output = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []
        memory_input = []
        memory_len_input = []
        input_len = []
        seg_memory_input = []
        memory_attention_mask_input = []
        memory_id0 = []
        memory_id1 = []
        memory_query = []
        memory_segment_query = []
        memory_len_query = []
        memory_attention_mask_query = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)


            cls_token = torch.LongTensor([2]).to(q_seg.device)
            memory_query.append(torch.cat((cls_token,q_item),dim=0))
            memory_segment_query.append(torch.zeros_like(torch.cat((cls_token,q_item),dim=0)))
            memory_len_query.append(torch.LongTensor([len(q_item)+1]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(torch.cat((cls_token,q_item),dim=0)))

            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            if q_start[i] != 1:
                memory = q_ids[i][:q_start[i]]
            else:
                memory = q_ids[i][:q_start[i]] 
                sep = torch.LongTensor([3]).to(q_seg.device)
                memory = torch.cat((memory,sep),dim=0)
            memory_len_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_attention_mask_input.append(torch.ones_like(memory))

            # sep token id  == 3
            sep_symbol = torch.nonzero((memory==3),as_tuple=False)
            
            if sep_symbol.shape[0] == 1:
                memory_init_seg = torch.zeros_like(memory)
            else:
                memory_init_seg_list = []
                start = 0
                for index in range(sep_symbol.squeeze(-1).shape[0]):
                    if index % 2 == 0:
                        memory_seg = torch.zeros_like(memory[start:sep_symbol[index]+1])
                    else:
                        memory_seg = torch.ones_like(memory[start:sep_symbol[index]+1])
                    start = sep_symbol[index]+1
                    memory_init_seg_list.append(memory_seg)
                memory_init_seg = torch.cat(memory_init_seg_list, dim=0)
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)



        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        memory_ids = pad_sequence(memory_input,batch_first=True)
        input_total_length = torch.stack(input_len).squeeze(1)
        memory_len = torch.stack(memory_len_input).squeeze(1)
        memory_attention_mask = pad_sequence(memory_attention_mask_input, batch_first=True)
        memory_segment = pad_sequence(seg_memory_input, batch_first=True)
        memory_segment_query = pad_sequence(memory_segment_query, batch_first=True)
        memory_length_query=torch.stack(memory_len_query)
        memory_query = pad_sequence(memory_query, batch_first=True)
        memory_attention_mask_query = pad_sequence(memory_attention_mask_query, batch_first=True)



        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
            "input_len":input_total_length,
            "memory": memory_ids,
            "memory_len":memory_len,
            "memory_attention_mask": memory_attention_mask,
            "memory_segment": memory_segment,
            "memory_query": memory_query,
            "memory_query_len":memory_len_query,
            "memory_segment_query":memory_segment_query,
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

class RobertaQA_memory(nn.Module):
    def __init__(self, config):
        super(RobertaQA_memory, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =RobertaConfig.from_pretrained("roberta-base")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size})
            self.pretrained_model = RobertaModelMemory2.from_pretrained("roberta-base", config=self.model_config)
        else:
            self.model_config =RobertaConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size})
            self.pretrained_model = RobertaModelMemory2.from_pretrained(config.pretrained_name, config=self.model_config)

        self.qa_output = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []
        memory_input = []
        memory_len_input = []
        input_len = []
        seg_memory_input = []
        memory_attention_mask_input = []
        memory_query = []
        memory_segment_query = []
        memory_len_query = []
        memory_attention_mask_query = []
        memory_ids0 = []
        memory_ids1 = []
        memory_ids2 = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(0)
            c_seg = torch.zeros_like(c_item).fill_(0)

            #current query
            cls_token= torch.LongTensor([0]).to(q_ids.device)
            memory_query.append(torch.cat((cls_token,q_item),dim=0))
            memory_segment_query.append(torch.zeros_like(torch.cat((cls_token,q_item),dim=0)))
            memory_len_query.append(torch.LongTensor([len(q_item)+1]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(torch.cat((cls_token,q_item),dim=0)))
            memory_ids2.append(torch.zeros_like(torch.cat((cls_token,q_item),dim=0)))

            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            memory_ids0.append(torch.zeros_like(torch.cat([c_item,q_item],dim=0)))



            if q_start[i] != 1:
                memory = q_ids[i][:q_start[i]]
            else:
                memory = q_ids[i][:q_start[i]] 
                sep = torch.LongTensor([2]).to(q_seg.device)
                memory = torch.cat((memory,sep),dim=0)
            memory_len_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_attention_mask_input.append(torch.ones_like(memory))

            memory_init_seg = torch.zeros_like(memory)
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)
            memory_ids1.append(torch.ones_like(memory))



        reason_mask= pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True, padding_value=1)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        input_total_length = torch.stack(input_len).squeeze(1)
        memory_ids0 = pad_sequence(memory_ids0,batch_first=True)

        memory_ids = pad_sequence(memory_input,batch_first=True,padding_value=1)
        memory_len = torch.stack(memory_len_input).squeeze(1)
        memory_attention_mask = pad_sequence(memory_attention_mask_input, batch_first=True)
        memory_segment = pad_sequence(seg_memory_input, batch_first=True)
        memory_ids1 = pad_sequence(memory_ids1,batch_first=True)


        memory_segment_query = pad_sequence(memory_segment_query, batch_first=True)
        memory_length_query=torch.stack(memory_len_query)
        memory_query = pad_sequence(memory_query, batch_first=True,padding_value=1)
        memory_attention_mask_query = pad_sequence(memory_attention_mask_query, batch_first=True)
        memory_ids2 = pad_sequence(memory_ids2,batch_first=True)




        input_args = {
            "input_ids": reason_ids,
            "attention_mask": reason_mask,
            "token_type_ids": reason_seg,
            "input_len":input_total_length,
            "memory": memory_ids,
            "memory_len": memory_len,
            "memory_attention_mask":memory_attention_mask,
            "memory_segment": memory_segment,
            "memory_query": memory_query,
            "memory_len_query":memory_len_query,
            "memory_segment_query":memory_segment_query,
            "memory_input_ids0":memory_ids0,
            "memory_input_ids1":memory_ids1,
            "memory_input_ids2":memory_ids2,        
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

class ALBERTQA(nn.Module):
    def __init__(self, config):
        super(ALBERTQA, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =AlbertConfig.from_pretrained("albert-base-v2")
            self.model_config.update(config.model_config)
            self.pretrained_model = AlbertModelRelative.from_pretrained("albert-base-v2",config=self.model_config)

        else:
            self.model_config =AlbertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.pretrained_model = AlbertModelRelative.from_pretrained(config.pretrained_name,config=self.model_config)
        
        self.qa_output = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            sep_symbol = torch.nonzero((q_item==3),as_tuple=False)

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


class BERTQA(nn.Module):
    def __init__(self, config):
        super(BERTQA, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.pretrained_model = BertModelRelative.from_pretrained("bert-base-uncased",config=self.model_config)

        else:
            self.model_config = BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.pretrained_model = BertModelRelative.from_pretrained(config.pretrained_name,config=self.model_config)
        
        self.qa_output = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
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

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output

class RobertaQA(nn.Module):
    def __init__(self, config):
        super(RobertaQA, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =RobertaConfig.from_pretrained("roberta-base")
            self.model_config.update(config.model_config)
            self.pretrained_model = RobertaModelRelative.from_pretrained("roberta-base",config=self.model_config)

        else:
            self.model_config = RobertaConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.pretrained_model = RobertaModelRelative.from_pretrained(config.pretrained_name,config=self.model_config)
        
        self.qa_output = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_segs, q_att_masks, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            # sep_symbol = torch.nonzero((q_item==3),as_tuple=False)

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
