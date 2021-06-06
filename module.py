import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, AlbertModelRelative, \
    RobertaModelRelative, BertConfig, BertModelRelative,\
    AlbertConfig, RobertaConfig, BertModelinitial_from_mu_sigma, \
    BertModelMemory2, AlbertModelMemory, RobertaModelMemory, RobertaModelMemory2,BertInjectMemory, BertModelreusemodule, BertModelMemory3, \
    BartForQuestionAnsweringMy, BartConfig, BertModelMemory4, BertModelMemory5, BertModelMemory6, BertModelMemory10, BertModelMemory11, BertModelMemory12, BertModelMemory115,BertModelMemory20, \
    BertModelMemory13, HisBERTModel, BertModelMemory14, BertModelMemory15, BertModelMemory16, BertModelMemory17, BertModelMemory18, BertModelRelativeHAE, BertModelMemory21, BertModelMemory22, BertModelMemory23, BertLMHeadModel

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
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


class BERTQA_memory_HAE(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory_HAE, self).__init__()
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelRelativeHAE.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelRelativeHAE.from_pretrained(config.pretrained_name, config=self.model_config)
        
        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)
        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends, **kwargs):

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
        history_id_input = []

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
            
            input_ids_tensor =torch.cat([c_item,q_item],dim=0)
            history_init = torch.zeros_like(input_ids_tensor)

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1
            history_id_input.append(history_init)
            
            
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


        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
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
            "history_ids": reason_history_ids, 
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

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output





class BERTQA_memory_HAE_future(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory_HAE_future, self).__init__()
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelRelativeHAE.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelRelativeHAE.from_pretrained(config.pretrained_name, config=self.model_config)
        
        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.qa_output_dialog_future = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)
        self.m = nn.Softmax(dim=1)
        self.m_ = nn.Softmax(dim=0)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends, future_starts, future_ends, **kwargs):

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
        history_id_input = []

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
            
            input_ids_tensor =torch.cat([c_item,q_item],dim=0)
            history_init = torch.zeros_like(input_ids_tensor)

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1
            history_id_input.append(history_init)
            
            
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


        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
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
            "history_ids": reason_history_ids, 
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        future_logits=self.qa_output_dialog_future(hidden_states)
        future_start_logits, future_end_logits = future_logits.split(1, dim=-1)
        future_start_logits = future_start_logits.squeeze(-1)
        future_end_logits = future_end_logits.squeeze(-1)


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

            if future_starts is not None and future_ends is not None:
    
                future_starts = future_starts.squeeze(-1)
                future_ends = future_ends.squeeze(-1)
                
                if len(future_starts.size()) > 1:
                    future_starts = future_starts.squeeze(-1)
                if len(end_positions.size()) > 1:
                    future_ends = future_ends.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = future_start_logits.size(1)
                future_starts.clamp_(0, ignored_index)
                future_ends.clamp_(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                future_start_loss = 0
                future_end_loss = 0
                number = 0
                for i in range(len(future_start_logits)):
                    if future_starts[i] == future_ends[i] == 0:
                        continue
                    number +=1
                    future_start_loss += loss_fct(future_start_logits[i][None,:], future_starts[i,None])
                    future_end_loss += loss_fct(future_end_logits[i][None, :], future_ends[i,None])
                if number >0:
                    future_start_loss /= number
                    future_end_loss /= number

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss + dialog_loss + future_start_loss + future_end_loss) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output





class BERTQA_memory_HAE_coqa(BERTQA_memory_HAE):
    def __init__(self, config):
        super(BERTQA_memory_HAE_coqa, self).__init__(config)
        #BERT layer, n_dim, nhead

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends, **kwargs):

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
        history_id_input = []

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
            
            input_ids_tensor =torch.cat([c_item,q_item],dim=0)
            history_init = torch.zeros_like(input_ids_tensor)

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1
            history_id_input.append(history_init)
            
            
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


        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
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
            "history_ids": reason_history_ids, 
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




        
class BERTQA_memory_coqa(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory_coqa, self).__init__(config)

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
        super(BERTQA_memory, self).__init__(config)

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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



class BERTQA_memoryHistory(nn.Module):
    def __init__(self, config):
        super(BERTQA_memoryHistory, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = HisBERTModel.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = HisBERTModel.from_pretrained(config.pretrained_name, config=self.model_config)

        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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



class BERTQA_memoryHistory_future(nn.Module):
    def __init__(self, config):
        super(BERTQA_memoryHistory_future, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = HisBERTModel.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = HisBERTModel.from_pretrained(config.pretrained_name, config=self.model_config)

        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.qa_output_dialog_future = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, future_starts, future_ends, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
        prediction=self.dialog_output(last_reps[1])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

        future_logits=self.qa_output_dialog_future(hidden_states)
        future_start_logits, future_end_logits = future_logits.split(1, dim=-1)
        future_start_logits = future_start_logits.squeeze(-1)
        future_end_logits = future_end_logits.squeeze(-1)

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

            if future_starts is not None and future_ends is not None:
        
                future_starts = future_starts.squeeze(-1)
                future_ends = future_ends.squeeze(-1)
                
                if len(future_starts.size()) > 1:
                    future_starts = future_starts.squeeze(-1)
                if len(end_positions.size()) > 1:
                    future_ends = future_ends.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = future_start_logits.size(1)
                future_starts.clamp_(0, ignored_index)
                future_ends.clamp_(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                future_start_loss = 0
                future_end_loss = 0
                number = 0
                for i in range(len(future_start_logits)):
                    if future_starts[i] == future_ends[i] == 0:
                        continue
                    number +=1
                    future_start_loss += loss_fct(future_start_logits[i][None,:], future_starts[i,None])
                    future_end_loss += loss_fct(future_end_logits[i][None, :], future_ends[i,None])
                if number >0:
                    future_start_loss /= number
                    future_end_loss /= number

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss + dialog_loss + future_start_loss +future_end_loss) /3

        output = (start_logits, end_logits, )
        return ((total_loss,) + output) if total_loss is not None else output




class BERTQA_memoryHistory_coqa(BERTQA_memoryHistory):
    def __init__(self, config):
        super(BERTQA_memoryHistory_coqa, self).__init__(config)
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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




class BERTQA_memory3(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory3, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory3.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory3.from_pretrained(config.pretrained_name, config=self.model_config)


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory4(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory4, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory4.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict())

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory4.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict())



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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




class BERTQA_memory5(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory5, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory5.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict())

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory5.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict())



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory6(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory6, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory6.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict(), strict=False)
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict(), strict=False)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory6.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict(), strict=False)
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict(), strict=False)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory10(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory10, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory10.from_pretrained("bert-base-uncased", config=self.model_config)


        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory10.from_pretrained(config.pretrained_name, config=self.model_config)




        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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




class BERTQA_memory11(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory11, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory11.from_pretrained("bert-base-uncased", config=self.model_config)


        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory11.from_pretrained(config.pretrained_name, config=self.model_config)




        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, *kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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




class BERTQA_memory115(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory115, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory115.from_pretrained("bert-base-uncased", config=self.model_config)


        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory115.from_pretrained(config.pretrained_name, config=self.model_config)




        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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





class BERTQA_memory12(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory12, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory12.from_pretrained("bert-base-uncased", config=self.model_config)


        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory12.from_pretrained(config.pretrained_name, config=self.model_config)




        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            #current query add context attention 
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory13(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory13, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory13.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict())

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory13.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_generator.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict())



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory14(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory14, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory14.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory14.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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

class BERTQA_memory14_coqa(BERTQA_memory14):
    def __init__(self, config):
        super(BERTQA_memory14_coqa, self).__init__(config)
        # memory attention on 1, 12, history module on 1, 12
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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





class BERTQA_memory15(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory15, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory15.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory15.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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



class BERTQA_memory20(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory20, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory20.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory20.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory21(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory21, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory21.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory21.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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





class BERTQA_memory21(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory21, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory21.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory21.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory22(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory22, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory22.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory22.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.prev_answer = nn.Linear(self.model_config.hidden_size, 2)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends, **kwargs):

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
        history_id_input = []
        history_doc_input = []
        history_mem_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            history_init = torch.zeros_like(torch.cat([c_seg,q_seg],dim=0))
            history_doc_init = torch.zeros_like(c_seg)

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1
                history_doc_init[history_starts[i][index]: history_ends[i][index]+1] = 1

            history_id_input.append(history_init)
            history_doc_input.append(history_doc_init)

            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
                sep = torch.LongTensor([102]).to(q_seg.device)
                memory = torch.cat((memory,sep),dim=0)
            memory_len_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_attention_mask_input.append(torch.ones_like(memory))

            history_mem_init = torch.zeros_like(memory)
            history_mem_input.append(history_mem_init)


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

        reason_history_mem_ids = pad_sequence(history_mem_input, batch_first=True)
        reason_history_doc_ids = pad_sequence(history_doc_input, batch_first=True)
        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
            "history_ids": reason_history_ids, 
            "history_doc_ids": reason_history_doc_ids,
            "history_mem_ids": reason_history_mem_ids,
            
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






class BERTQA_memory23(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory23, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory23.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_mem.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings_mem.state_dict()) 

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory23.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_mem.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict()) 


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()
        self.prev_answer= nn.Linear(self.model_config.hidden_size, 2)


    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends,**kwargs):

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
        history_id_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            history_init = torch.zeros_like(torch.cat([c_seg,q_seg],dim=0))

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1

            history_id_input.append(history_init)
            
            memory = q_ids[i][:q_start[i]]
            #memory = q_ids[i][:q_len[i]]
            if q_start[i] == 1:
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
                #memory_init_seg_list = []
                #memory_init_seg = torch.zeros_like(memory)
                #memory_init_seg[q_start[i]:] = 1
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)
        
        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
        prediction=self.dialog_output(hidden_states[:,0])
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
            total_loss = (start_loss + end_loss) + (dialog_loss) / 2

        output = (start_logits, end_logits, )
        return ((total_loss,) + output) if total_loss is not None else output





class BERTQA_memory23_future(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory23_future, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory23.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_mem.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings_mem.state_dict()) 

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory23.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_mem.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict()) 


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.qa_output_dialog_future = nn.Linear(self.model_config.hidden_size, config.num_labels)
        # self.transform = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()
        # self.act_fn=nn.GELU()
        self.softmax = nn.Softmax(dim=-2)
        self.prev_answer= nn.Linear(self.model_config.hidden_size, 2)


    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends, future_starts, future_ends, **kwargs):

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
        history_id_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            history_init = torch.zeros_like(torch.cat([c_seg,q_seg],dim=0))

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1

            history_id_input.append(history_init)
            
            memory = q_ids[i][:q_start[i]]
            #memory = q_ids[i][:q_len[i]]
            if q_start[i] == 1:
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
                #memory_init_seg_list = []
                #memory_init_seg = torch.zeros_like(memory)
                #memory_init_seg[q_start[i]:] = 1
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)
        
        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
        prediction=self.dialog_output(hidden_states[:,0])
        dialog_loss = self.dialog_loss_fct(prediction, dialog_act)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        future_logits=self.qa_output_dialog_future(hidden_states)
        future_start_logits, future_end_logits = future_logits.split(1, dim=-1)
        future_start_logits = future_start_logits.squeeze(-1)
        future_end_logits = future_end_logits.squeeze(-1)


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

            if future_starts is not None and future_ends is not None:
    
                future_starts = future_starts.squeeze(-1)
                future_ends = future_ends.squeeze(-1)
                
                if len(future_starts.size()) > 1:
                    future_starts = future_starts.squeeze(-1)
                if len(end_positions.size()) > 1:
                    future_ends = future_ends.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = future_start_logits.size(1)
                future_starts.clamp_(0, ignored_index)
                future_ends.clamp_(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                future_start_loss = 0
                future_end_loss = 0
                number = 0
                for i in range(len(future_start_logits)):
                    if future_starts[i] == future_ends[i] == 0:
                        continue
                    number +=1
                    future_start_loss += loss_fct(future_start_logits[i][None,:], future_starts[i,None])
                    future_end_loss += loss_fct(future_end_logits[i][None, :], future_ends[i,None])
                if number >0:
                    future_start_loss /= number
                    future_end_loss /= number

                # future_start_loss = loss_fct(future_start_logits, future_starts)
                # future_end_loss = loss_fct(future_end_logits, future_ends)   

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)           

            total_loss = ((start_loss + end_loss) + (dialog_loss) + (future_start_loss + future_end_loss)) / 3

        output = (start_logits, end_logits, )
        return ((total_loss,) + output) if total_loss is not None else output




class BERTQA_memory23_question(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory23_question, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory23.from_pretrained("bert-base-uncased", config=self.model_config)
            self.pretrained_model.encoder_mem.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings_mem.state_dict()) 

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory23.from_pretrained(config.pretrained_name, config=self.model_config)
            self.pretrained_model.encoder_mem.load_state_dict(self.pretrained_model.encoder.state_dict())
            self.pretrained_model.embeddings_mem.load_state_dict(self.pretrained_model.embeddings.state_dict()) 


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-2)
        self.prev_answer= nn.Linear(self.model_config.hidden_size, 2)
        self.pretrained_decoder_config = BertConfig.from_pretrained(config.pretrained_name)
        self.pretrained_decoder_config.is_decoder=True
        self.pretrained_decoder_config.add_cross_attention=True
        self.pretrained_decoder_config.num_hidden_layers = 3
        self.pretrained_decoder = BertLMHeadModel.from_pretrained("bert-base-uncased", config=self.pretrained_decoder_config)

        # self.decoder = 


    def forward(self, q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions,history_starts, history_ends, future_starts, future_ends,future_q, future_att, **kwargs):

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
        history_id_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            history_init = torch.zeros_like(torch.cat([c_seg,q_seg],dim=0))

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1

            history_id_input.append(history_init)
            
            memory = q_ids[i][:q_start[i]]
            #memory = q_ids[i][:q_len[i]]
            if q_start[i] == 1:
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
                #memory_init_seg_list = []
                #memory_init_seg = torch.zeros_like(memory)
                #memory_init_seg[q_start[i]:] = 1
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)
        
        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
        prediction=self.dialog_output(hidden_states[:,0])
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

            #==============================================
            input_args = {
                "input_ids":future_q,
                "attention_mask":future_att,
                "encoder_hidden_states":hidden_states,
                "encoder_attention_mask":reason_mask,
                "labels":future_q,
                "return_dict":False,
                }
                
            output = self.pretrained_decoder(**input_args)
            lm_loss = output[0]
            
            total_loss = (start_loss + end_loss) + (dialog_loss) + lm_loss / 3

        output = (start_logits, end_logits, )
        return ((total_loss,) + output) if total_loss is not None else output




class BERTQA_memory23_coqa(BERTQA_memory23):
    def __init__(self, config):
        super(BERTQA_memory23_coqa, self).__init__(config)

        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)


    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends,**kwargs):

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
        history_id_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            history_init = torch.zeros_like(torch.cat([c_seg,q_seg],dim=0))

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1

            history_id_input.append(history_init)
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
        }


        last_reps = self.pretrained_model(**input_args)

        hidden_states = last_reps[0]

        logits=self.qa_output_dialog(hidden_states)
        prediction=self.dialog_output(hidden_states[:,0])
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
            total_loss = (start_loss + end_loss) + (dialog_loss) / 2

        output = (start_logits, end_logits, prediction)
        return ((total_loss,) + output) if total_loss is not None else output

class BERTQA_memory22_coqa(BERTQA_memory22):
    def __init__(self, config):
        super(BERTQA_memory22_coqa, self).__init__(config)
        # interact module over and over again
        #BERT layer, n_dim, nhead
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, history_starts, history_ends, **kwargs):

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
        history_id_input = []
        history_doc_input = []
        history_mem_input = []


        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([101]).to(q_seg.device)
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))

            history_init = torch.cat([c_seg,q_seg],dim=0)
            history_doc_init = torch.zeros_like(c_seg)
            history_mem_init = torch.ones_like(q_seg)

            for index in range(len(history_starts[i])):
                if history_starts[i][index] == 0:
                    continue
                history_init[history_starts[i][index]: history_ends[i][index]+1] = 1
                history_doc_init[history_starts[i][index]: history_ends[i][index]+1] = 1

            history_id_input.append(history_init)
            history_doc_input.append(history_doc_init)
            history_mem_input.append(history_mem_init)

            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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

        reason_history_mem_ids = pad_sequence(history_mem_input, batch_first=True)
        reason_history_doc_ids = pad_sequence(history_doc_input, batch_first=True)
        reason_history_ids = pad_sequence(history_id_input, batch_first=True)
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
            "history_ids": reason_history_ids, 
            "history_doc_ids": reason_history_doc_ids,
            "history_mem_ids": reason_history_mem_ids,
            
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





class BERTQA_memory15_coqa(BERTQA_memory15):
    def __init__(self, config):
        super(BERTQA_memory15_coqa, self).__init__(config)
        #BERT layer, n_dim, nhead
        # layer 1, 12 with memory attention, but history on 0, 12
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)


    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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

class BERTQA_memory16(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory16, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory16.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory16.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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





class BERTQA_memory16_coqa(BERTQA_memory16):
    def __init__(self, config):
        super(BERTQA_memory16_coqa, self).__init__(config)
        # interact module over and over again
        #BERT layer, n_dim, nhead
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)


    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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





class BERTQA_memory17(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory17, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory17.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory17.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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


class BERTQA_memory17_coqa(BERTQA_memory17):
    def __init__(self, config):
        super(BERTQA_memory17_coqa, self).__init__(config)
        # interact module over and over again
        #BERT layer, n_dim, nhead
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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






class BERTQA_memory18(nn.Module):
    def __init__(self, config):
        super(BERTQA_memory18, self).__init__()
        # interact module over and over again
        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory18.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelMemory18.from_pretrained(config.pretrained_name, config=self.model_config)



        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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





class BERTQA_memory18_coqa(BERTQA_memory18):
    def __init__(self, config):
        super(BERTQA_memory18_coqa, self).__init__()
        
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            memory_query.append(c_item)
            memory_segment_query.append(torch.zeros_like(c_seg))
            memory_len_query.append(torch.LongTensor([len(c_item)]).to(q_seg.device))
            memory_attention_mask_query.append(torch.ones_like(c_item))
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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

class BERTQA_memoryreuse(nn.Module):
    def __init__(self, config):
        super(BERTQA_memoryreuse, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BertConfig.from_pretrained("bert-base-uncased")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelreusemodule.from_pretrained("bert-base-uncased", config=self.model_config)

        else:
            self.model_config =BertConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BertModelreusemodule.from_pretrained(config.pretrained_name, config=self.model_config)


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
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
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
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
            "query_att_mask":memory_attention_mask_query,
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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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


class BERTQA_future(nn.Module):
    def __init__(self, config):
        super(BERTQA_future, self).__init__()

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
        self.qa_output_future = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, future_starts, future_ends, **kwargs):

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

        future_logits=self.qa_output_future(last_reps[0])
        future_start_logits, future_end_logits = future_logits.split(1, dim=-1)
        future_start_logits = future_start_logits.squeeze(-1)
        future_end_logits = future_end_logits.squeeze(-1)


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

            if future_starts is not None and future_ends is not None:
        
                future_starts = future_starts.squeeze(-1)
                future_ends = future_ends.squeeze(-1)
                
                if len(future_starts.size()) > 1:
                    future_starts = future_starts.squeeze(-1)
                if len(end_positions.size()) > 1:
                    future_ends = future_ends.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = future_start_logits.size(1)
                future_starts.clamp_(0, ignored_index)
                future_ends.clamp_(0, ignored_index)

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                future_start_loss = 0
                future_end_loss = 0
                number = 0
                for i in range(len(future_start_logits)):
                    if future_starts[i] == future_ends[i] == 0:
                        continue
                    number +=1
                    future_start_loss += loss_fct(future_start_logits[i][None,:], future_starts[i,None])
                    future_end_loss += loss_fct(future_end_logits[i][None, :], future_ends[i,None])
                if number >0:
                    future_start_loss /= number
                    future_end_loss /= number


            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss  + dialog_loss + future_start_loss + future_end_loss) /3

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output



class BERTQA_original(nn.Module):
    def __init__(self, config):
        super(BERTQA_original, self).__init__()

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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
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



class BERTQA_original_coqa(BERTQA_original):
    def __init__(self, config):
        super(BERTQA_original_coqa, self).__init__(config)
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

        reason_input = []
        seg_input = []
        mask_input = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
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






class BERTQA_coqa(BERTQA):
    def __init__(self, config):
        super(BERTQA_coqa, self).__init__(config)
        self.dialog_loss_fct = nn.CrossEntropyLoss(ignore_index=-1000)

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

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

        output = (start_logits, end_logits, prediction)
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

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, *kwargs):

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



class BARTQA(nn.Module):
    def __init__(self, config):
        super(BARTQA, self).__init__()

        #BERT layer, n_dim, nhead
        config = Namespace(**config)

        if "pretrained_name" not in list(vars(config).keys()):
            self.model_config =BartConfig.from_pretrained("facebook/bart-base")
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BartForQuestionAnsweringMy.from_pretrained("facebook/bart-base", config=self.model_config)

        else:
            self.model_config =BartConfig.from_pretrained(config.pretrained_name)
            self.model_config.update(config.model_config)
            self.model_config.update({"bottleneck_size": config.bottleneck_size, "add_cross_attention":config.add_cross_attention})
            self.pretrained_model = BartForQuestionAnsweringMy.from_pretrained(config.pretrained_name, config=self.model_config)


        self.qa_output_dialog = nn.Linear(self.model_config.hidden_size, config.num_labels)
        self.dialog_output = nn.Linear(self.model_config.hidden_size, config.dialog_labels)
        self.dialog_loss_fct = nn.CrossEntropyLoss()

    def forward(self,q_ids, q_start, q_len, c_ids, c_att_masks, c_len, dialog_act,start_positions, end_positions, **kwargs):

        reason_input = []
        seg_input = []
        mask_input = []
        memory_input = []
        memory_len_input = []
        input_len = []
        seg_memory_input = []
        memory_attention_mask_input = []
        memory_position_id = []

        for i in range(len(q_ids)):
            q_item = q_ids[i][q_start[i]:q_len[i]]
            c_item = c_ids[i][:c_len[i]]

            q_seg = torch.zeros_like(q_item).fill_(1)
            c_seg = torch.zeros_like(c_item).fill_(0)
            
            #current query
            cls_token = torch.LongTensor([0]).to(q_seg.device)
            
            seg_input.append(torch.cat([c_seg,q_seg],dim=0))
            mask_input.append(torch.ones_like(torch.cat([c_seg,q_seg],dim=0)))
            reason_input.append(torch.cat([c_item,q_item],dim=0))
            input_len.append(torch.LongTensor([len(torch.cat([c_seg,q_seg],dim=0))]))
            
            memory = q_ids[i][:q_start[i]]
            if q_start[i] == 1:
                sep = torch.LongTensor([2]).to(q_seg.device)
                memory = torch.cat((memory,sep),dim=0)
            memory_len_input.append(torch.LongTensor([len(memory)]).to(q_seg.device))
            memory_attention_mask_input.append(torch.ones_like(memory))

            # sep token id  == 102 
            sep_symbol = torch.nonzero((memory==2),as_tuple=False)
            
            if sep_symbol.shape[0] == 1:
                memory_init_seg = torch.zeros_like(memory)
            else:
                memory_init_seg_list = []
                start = 0
                for index in range(0, sep_symbol.squeeze(-1).shape[0],2):
                    if index % 2 == 0:
                        memory_seg = torch.zeros_like(memory[start:sep_symbol[index]+1])
                    else:
                        memory_seg = torch.ones_like(memory[start:sep_symbol[index]+1])
                    start = sep_symbol[index]+1
                    memory_init_seg_list.append(memory_seg)
                memory_init_seg = torch.cat(memory_init_seg_list, dim=0)
            seg_memory_input.append(memory_init_seg)
            memory_input.append(memory)
        
        reason_mask = pad_sequence(mask_input, batch_first=True)
        reason_ids = pad_sequence(reason_input, batch_first=True)
        reason_seg = pad_sequence(seg_input,batch_first=True)
        memory_ids = pad_sequence(memory_input,batch_first=True)
        input_total_length = torch.stack(input_len).squeeze(1)
        memory_len = torch.stack(memory_len_input).squeeze(1)
        memory_segment = pad_sequence(seg_memory_input, batch_first=True)
        memory_attention_mask = pad_sequence(memory_attention_mask_input, batch_first=True)
        

        input_args = {
            "input_ids": memory_ids,
            "attention_mask": memory_attention_mask,
            "decoder_input_ids": reason_ids,
            "decoder_attention_mask":reason_mask,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

        return self.pretrained_model(**input_args, return_dict=False)
        
