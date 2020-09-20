import numpy
import torch
import IPython, pdb
import functools
import tqdm
import wandb
import json
import copy
from datasets import load_dataset, list_datasets, load_metric
from transformers import BertModel, BertConfig, BertTokenizerFast
from torch import nn
import logging
import torch_optimizer as optim
logging.basicConfig(level=logging.INFO)

from module_custom import TransformerDecoder, TransformerDecoderLayer, TransformerDecoderLayer_bmask, generate_square_subsequent_mask, get_linear_schedule_with_warmup



def example_to_utilize_BERT():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
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

def get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()

# Tokenize our training dataset
def convert_to_features(example_batch, max_length, stride):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer(example_batch['context'], truncation=True, max_length=max_length, padding="max_length", stride=stride, return_length=True)
    example_batch_answer = [answer['text'][0] for answer in example_batch['answers']]
    question_ans_encodings = tokenizer(example_batch['question'], example_batch_answer, truncation=True, max_length=30,padding="max_length", return_length=True)
    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methods.
    start_positions, end_positions = [], []
    for i, (context, answer) in enumerate(zip(example_batch['context'], example_batch['answers'])):
        start_idx, end_idx = get_correct_alignement(context, answer)
        start_positions.append(encodings.char_to_token(i, start_idx))
        end_positions.append(encodings.char_to_token(i, end_idx-1))
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    encodings.update({"qans_input_ids":question_ans_encodings['input_ids'], "qans_token_type_ids":question_ans_encodings['token_type_ids'], "qans_att_mask": question_ans_encodings['attention_mask'], "qans_length":question_ans_encodings['length']})
    return encodings

# Tokenize our eval dataset
def convert_to_eval_features(example_batch, max_length, stride):
    # Tokenize contexts and questions (as pairs of inputs)
    encodings = tokenizer(example_batch['context'], truncation=False, max_length=max_length, padding="max_length", stride=stride, return_length=True)
    question_ans_encodings = tokenizer([example_batch['question']], truncation=False, return_length=True)
    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methods.
    encodings.update({"qans_input_ids":question_ans_encodings['input_ids'], "qans_token_type_ids":question_ans_encodings['token_type_ids'], "qans_att_mask": question_ans_encodings['attention_mask'], "qans_length":question_ans_encodings['length']})
    return encodings

def collate_fn(examples):
    return tokenizer.pad(examples, return_tensors='pt')

def collate_fn_eval(examples):
    return tokenizer.pad(examples)

if __name__ == "__main__":
    
    #config
    epoches = 6
    max_length = 512
    stride = 128
    hid_size = 512
    n_head = 16
    n_layer = 6
    layer_eps = 1e-12
    batch_size = 32
    lr = 2e-4
    log_step = 100
    dev_step = 500
    save_step = 1000
    warm_up_ratio = 0.2
    gradient_accumulation_steps = 5
    gradient_clipping = 1.0
    eval_file_path = "eval_squad.json"
    exp_name = "plot_training_loss"

    wandb.init(project="Response-writer", name=exp_name)

    # setup dataset, dataloader, map function
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    squad_dataset = load_dataset("squad", cache_dir="./squad")
    squad_dataset_eval = load_dataset("squad", cache_dir="./squad", split="validation")
    convert_to_features =functools.partial(convert_to_features, max_length=max_length, stride=stride)
    squad_dataset= squad_dataset.map(convert_to_features,batched=True)
    convert_to_eval_features =functools.partial(convert_to_eval_features, max_length=max_length, stride=stride)
    squad_dataset_eval = squad_dataset_eval.map(convert_to_eval_features,batched=False)

    #load pretrained model (using pretrained vector)
    model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    # special_tokens_dict = {"additional_special_tokens":['<Q>','<ans>','<eos>']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

    decoder_layer = TransformerDecoderLayer(d_model=hid_size, nhead=n_head)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=n_layer)
    projector = nn.Linear(768, hid_size)
    project_norm = nn.LayerNorm(hid_size, eps=layer_eps)
    inv_projector = nn.Linear(hid_size,768)
    act_fn = nn.functional.gelu

    # Format our dataset to outputs torch.Tensor and just take below attribute out
    columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'length', "qans_input_ids","qans_token_type_ids", "qans_att_mask", "qans_length"]
    squad_dataset.set_format(type='torch',columns=columns)
    columns_eval = ['input_ids', 'token_type_ids', 'attention_mask', 'length', "qans_input_ids","qans_token_type_ids", "qans_att_mask", "qans_length","id","answers"]
    squad_dataset_eval.set_format(type=None,columns=columns_eval)

    #setting dataloader, optimizer, scheduler
    dataloader = torch.utils.data.DataLoader(squad_dataset['train'], collate_fn=collate_fn, batch_size=batch_size)
    dev_dataloader = torch.utils.data.DataLoader(squad_dataset['validation'], collate_fn=collate_fn, batch_size=batch_size)
    dev_dataloader_gen = torch.utils.data.DataLoader(squad_dataset_eval, collate_fn=collate_fn_eval, batch_size=1)

    all_param = list(project_norm.parameters())+list(project_norm.parameters())+list(transformer_decoder.parameters())+list(inv_projector.parameters())
    # optimizer = torch.optim.AdamW(all_param, lr=lr)
    optimizer = optim.Lamb(all_param, lr=lr,betas=(0.9, 0.999),eps=1e-8,weight_decay=0,)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warm_up_ratio*len(dataloader)*epoches/gradient_accumulation_steps),num_training_steps=int(len(dataloader)*epoches/gradient_accumulation_steps))
    
    model.embeddings.word_embeddings.required_grad = False
    
    # put module to device (GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    transformer_decoder = transformer_decoder.to(device)
    project_norm = project_norm.to(device)
    projector = projector.to(device)
    inv_projector = inv_projector.to(device)

    # training-loss module
    lgsoft=torch.nn.LogSoftmax(dim=-1)
    loss_fn = nn.NLLLoss()

    #init training log
    logging_loss = 0
    train_loss = 0
    global_step = 0
    display_loss = 0
    step =0
    dev_loss = 0

    globalstep_bar = tqdm.tqdm(total=int(len(dataloader)*epoches/gradient_accumulation_steps),position=1)
    progress = tqdm.tqdm(dataloader, dynamic_ncols=True,position=2)
    progress_dev = tqdm.tqdm(dev_dataloader, dynamic_ncols=True,position=3)
    progress_eval = tqdm.tqdm(dev_dataloader_gen, dynamic_ncols=True,position=4)

    for epoch in tqdm.trange(epoches, dynamic_ncols=True,position=0):
        for num, batch in enumerate(progress):
            batch = batch.to(device)
            step +=1
            # generate input which need utilize BERT to generate pretrained embeddings
            encoder_input = {"input_ids":batch["input_ids"],"token_type_ids":batch["token_type_ids"],"attention_mask":batch["attention_mask"]}
            decoder_input = {"input_ids":batch["qans_input_ids"], "token_type_ids":batch["qans_token_type_ids"],"attention_mask":batch["qans_att_mask"],"output_hidden_states":True}

            with torch.no_grad():
                #load the last layer embedding from BERT
                memory_embed =model(**encoder_input)["last_hidden_state"]
                project_mem=projector(memory_embed)
                project_mem=project_norm(project_mem).transpose(0,1)
                
                #load the first layer embedding from BERT 
                decoder_embed = model(**decoder_input)['hidden_states'][0]

            #project to fix dimension    
            decoder_embed=projector(decoder_embed)
            decoder_embed=project_norm(decoder_embed)

            #generate casual masking
            tgt_mask = generate_square_subsequent_mask(decoder_embed.shape[1]) 
            decoder_embed=decoder_embed.transpose(0,1)

            #generate padding masking (attention should not attend on <pad> token)
            tgt_key_padding_mask = ~decoder_input["attention_mask"].bool()
            mem_key_padding_mask = ~encoder_input["attention_mask"].bool()

            forward_input = {"tgt":decoder_embed,"memory":project_mem,"tgt_mask":tgt_mask.to(device), "memory_key_padding_mask":mem_key_padding_mask, "tgt_key_padding_mask": tgt_key_padding_mask }

            #project to fix dimension    
            output = transformer_decoder(**forward_input).transpose(0, 1)
            inv_projector(output)
            output = inv_projector(output)
            output = act_fn(output)

            logits = torch.matmul(output, model.embeddings.word_embeddings.weight.T)
            log_distr = lgsoft(logits)

            # next input is the label of current input (shift left 1)
            labels = torch.roll(decoder_input["input_ids"], shifts=(0,-1),dims=(0,1))

            # remove the last element (meaningless)
            index_select = torch.nonzero(labels[:,:-2],as_tuple=False)

            # pick label whose token is not <pad>
            real_labels = labels[index_select[:,0], index_select[:,1]]

            # pick distribution whose token is not <pad>
            real_distr = log_distr[index_select[:,0], index_select[:,1],:]

            loss = loss_fn(real_distr,real_labels)
            loss /= gradient_accumulation_steps
            loss.backward()

            train_loss += loss.item()

            if (step % gradient_accumulation_steps) == 0:
                
                nn.utils.clip_grad_norm_(all_param,gradient_clipping)
                optimizer.step()
                scheduler.step()
                transformer_decoder.zero_grad()
                projector.zero_grad()
                inv_projector.zero_grad()
                global_step +=1
                globalstep_bar.update(1)

                if (global_step % log_step) == 0:
                    display_loss = (train_loss - logging_loss) / log_step
                    progress.set_description(f"Loss:{display_loss:.5f}")
                    wandb.log({"tr_loss": display_loss, "epoch": float(epoch+(num/len(dataloader)))}, step=global_step)
                    logging_loss = train_loss

                if (global_step % dev_step) == 0:
                    transformer_decoder = transformer_decoder.eval()
                    project_norm = project_norm.eval()
                    inv_projector = inv_projector.eval()
                    print("Start Dev")
                    for dev_i, batch in enumerate(progress_dev):
                        batch = batch.to(device)
                        # generate input which need utilize BERT to generate pretrained embeddings
                        encoder_input = {"input_ids":batch["input_ids"],"token_type_ids":batch["token_type_ids"],"attention_mask":batch["attention_mask"]}
                        decoder_input = {"input_ids":batch["qans_input_ids"], "token_type_ids":batch["qans_token_type_ids"],"attention_mask":batch["qans_att_mask"],"output_hidden_states":True}

                        with torch.no_grad():
                            #load the last layer embedding from BERT
                            memory_embed =model(**encoder_input)["last_hidden_state"]
                            project_mem=projector(memory_embed)
                            project_mem=project_norm(project_mem).transpose(0,1)
                            
                            #load the first layer embedding from BERT 
                            decoder_embed = model(**decoder_input)['hidden_states'][0]

                        #project to fix dimension    
                        decoder_embed=projector(decoder_embed)
                        decoder_embed=project_norm(decoder_embed)

                        #generate casual masking
                        tgt_mask = generate_square_subsequent_mask(decoder_embed.shape[1]) 
                        decoder_embed=decoder_embed.transpose(0,1)

                        #generate padding masking (attention should not attend on <pad> token)
                        tgt_key_padding_mask = ~decoder_input["attention_mask"].bool()
                        mem_key_padding_mask = ~encoder_input["attention_mask"].bool()

                        forward_input = {"tgt":decoder_embed,"memory":project_mem,"tgt_mask":tgt_mask.to(device), "memory_key_padding_mask":mem_key_padding_mask, "tgt_key_padding_mask": tgt_key_padding_mask }

                        #project to fix dimension    
                        output = transformer_decoder(**forward_input).transpose(0, 1)
                        inv_projector(output)
                        output = inv_projector(output)
                        output = act_fn(output)

                        logits = torch.matmul(output, model.embeddings.word_embeddings.weight.T)
                        log_distr = lgsoft(logits)

                        # next input is the label of current input (shift left 1)
                        labels = torch.roll(decoder_input["input_ids"], shifts=(0,-1),dims=(0,1))

                        # remove the last element (meaningless)
                        index_select = torch.nonzero(labels[:,:-2],as_tuple=False)

                        # pick label whose token is not <pad>
                        real_labels = labels[index_select[:,0], index_select[:,1]]

                        # pick distribution whose token is not <pad>
                        real_distr = log_distr[index_select[:,0], index_select[:,1],:]

                        eval_loss = loss_fn(real_distr,real_labels)
                        eval_loss /= gradient_accumulation_steps
                        dev_loss += eval_loss.item()
                    
                    print(f"dev loss: {dev_loss/(len(dev_dataloader)//gradient_accumulation_steps):.5f}")
                    wandb.log({"Dev loss": dev_loss}, step=global_step)
                    dev_loss = 0

                    if (global_step % save_step) == 0:
                        print("Start Save")
                        squad_metric = load_metric('squad')
                        answer_dict = {}
                        metric_dict = {}
                        ref_dict = {}
                        answer_list = []
                        ref_list = []
                        for eval_i, batch in enumerate(progress_eval):
                            encoder_input = {"input_ids":torch.tensor(batch['input_ids']).to(device),"attention_mask":torch.tensor(batch["attention_mask"]).to(device),"token_type_ids":torch.tensor(batch["token_type_ids"]).to(device)}
                            decoder_input = {"input_ids":torch.tensor(batch["qans_input_ids"]).squeeze(0).to(device), "token_type_ids":torch.tensor(batch["qans_token_type_ids"]).squeeze(0).to(device),"attention_mask":torch.tensor(batch["qans_att_mask"]).squeeze(0).to(device),"output_hidden_states":True}

                            with torch.no_grad():
                                #load the last layer embedding from BERT
                                memory_embed =model(**encoder_input)["last_hidden_state"]
                                project_mem=projector(memory_embed)
                                project_mem=project_norm(project_mem).transpose(0,1)
                                
                                #load the first layer embedding from BERT 
                                decoder_embed = model(**decoder_input)['hidden_states'][0]
                        
                            #project to fix dimension    
                            decoder_embed=projector(decoder_embed)
                            decoder_embed=project_norm(decoder_embed)

                            #generate casual masking
                            tgt_mask = generate_square_subsequent_mask(decoder_embed.shape[1]) 
                            decoder_embed=decoder_embed.transpose(0,1)

                            #generate padding masking (attention should not attend on <pad> token)
                            tgt_key_padding_mask = ~decoder_input["attention_mask"].bool()
                            mem_key_padding_mask = ~encoder_input["attention_mask"].bool()

                            max_loop = 20
                            loop_step = 0
                            next_word = ""
                            while next_word != "[SEP]" and loop_step < max_loop:
                                forward_input = {"tgt":decoder_embed,"memory":project_mem,"tgt_mask":tgt_mask.to(device), "memory_key_padding_mask":mem_key_padding_mask, "tgt_key_padding_mask": tgt_key_padding_mask }
                                #project to fix dimension    
                                output = transformer_decoder(**forward_input).transpose(0, 1)
                                inv_projector(output)
                                output = inv_projector(output)
                                output = act_fn(output)[:,-1,:]
                                next_id = torch.argmax(torch.matmul(output, model.embeddings.word_embeddings.weight.T))
                                next_word = tokenizer.convert_ids_to_tokens(next_id.tolist())
                                decoder_input["input_ids"] = torch.cat((decoder_input["input_ids"],next_id.unsqueeze(0).unsqueeze(0).to(device)), dim=1)
                                decoder_input["token_type_ids"] = torch.cat((decoder_input["token_type_ids"],torch.tensor([1]).unsqueeze(0).to(device)), dim=1)
                                decoder_input["attention_mask"] = torch.cat((decoder_input["attention_mask"],torch.tensor([1]).unsqueeze(0).to(device)), dim=1)


                                with torch.no_grad():
                                    #load the first layer embedding from BERT 
                                    decoder_embed = model(**decoder_input)['hidden_states'][0]

                                #project to fix dimension    
                                decoder_embed=projector(decoder_embed)
                                decoder_embed=project_norm(decoder_embed)

                                #generate casual masking
                                tgt_mask = generate_square_subsequent_mask(decoder_embed.shape[1]) 
                                decoder_embed=decoder_embed.transpose(0,1)
                                
                                #generate padding masking (attention should not attend on <pad> token)
                                tgt_key_padding_mask = ~decoder_input["attention_mask"].bool()

                                loop_step +=1
                            loop_step = 0
                            output_answer_string = tokenizer.decode(decoder_input["input_ids"][:,decoder_input["input_ids"].tolist()[0].index(102):].tolist()[0],skip_special_tokens=True)
                            answer_dict[batch["id"][0]] = output_answer_string
                            metric_dict["id"] = batch["id"][0]
                            metric_dict["prediction_text"] = output_answer_string
                            ref_dict["id"] = batch["id"][0]
                            ref_dict["answers"] = {"text":batch["answers"][0]['text'], "answer_start":batch["answers"][0]['answer_start']}
                            answer_list.append(copy.deepcopy(metric_dict))
                            ref_list.append(copy.deepcopy(ref_dict))
                        score = squad_metric.compute(predictions=answer_list,  references=ref_list)
                        # eval_json = json.dumps(answer_dict)
                        json.dump(answer_dict,open(eval_file_path,"w"))
                        print(f"EM: {score['exact_match']}, f1: {score['f1']}")
                        wandb.log({"eval_EM": score['exact_match'], "eval_F1":score['f1']}, step=global_step)
                    transformer_decoder = transformer_decoder.train()
                    project_norm = project_norm.train()
                    inv_projector = inv_projector.train()
