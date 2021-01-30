import os
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
import apex
from apex import amp
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils import set_seed
from optimizer_utils import *
from extract_feature_quac import *
from evaluate_utils_quac import *


def train(model, cache_train_file, cache_validation_file, eval_json, train_args, tokenizer, wandb):

    model = model.to(train_args['device'])

    preprocess = torch.load(cache_train_file)
    train_feature, train_dataset, train_examples = preprocess['features'], preprocess['dataset'], preprocess['examples']

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_args['batch_size'])

    no_decay = ["bias", "LayerNorm.weight"]

    t_total = train_args['epoches'] * len(train_dataloader) // train_args['gradient_accumulation_steps'] 

    optimizer_grouped_parameters = [
        {
            "lr": train_args['learning_rate']*1.0,"params": [p for n, p in model.named_parameters() if ("memory_module" in n or "dialog" in n or "embeddings" in n)],
            "weight_decay": train_args['weight_decay'],
        },
                {
            "lr": train_args['learning_rate']*1.0,"params": [p for n, p in model.named_parameters() if ("memory_module" not in n and "dialog" not in n and "embeddings" not in n)],
            "weight_decay": train_args['weight_decay'],
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, betas=(train_args['adam_beta1'],train_args["adam_beta2"]),lr=train_args['learning_rate'], eps=train_args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=train_args['warmup_steps'], num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_args['epoches'])
    logger.info("  Instantaneous batch size per GPU = %d", train_args['batch_size'])
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_args['batch_size']
        * train_args['gradient_accumulation_steps']
        * (torch.distributed.get_world_size() if train_args['local_rank'] != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", train_args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(train_args['epoches']), desc="Epoch", disable=train_args['local_rank'] not in [-1, 0]
    )

    BEST_F1 = np.array([0.0])
    BEST_STEP = np.array([0])

    if train_args['fp16']:
        scaler = torch.cuda.amp.GradScaler()


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=train_args['local_rank'] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(train_args['device']) for t in batch)

            inputs = {
                "c_ids": batch[0],
                "c_att_masks": batch[1],
                "c_len": batch[2],
                "q_ids": batch[3],
                "q_segs":batch[4],
                "q_att_masks": batch[5],
                "q_len": batch[6],
                "q_start": batch[7],
                "dialog_act": batch[8],
                "start_positions":batch[9],
                "end_positions": batch[10],
            }

            if train_args['fp16']:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    #model outputs are always tuple in transformers (see doc)
                    loss = outputs[0]
            else:
                outputs = model(**inputs)
                #model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

            if train_args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if train_args['gradient_accumulation_steps'] > 1:
                loss = loss / train_args['gradient_accumulation_steps']

            if train_args['fp16']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % train_args['gradient_accumulation_steps'] == 0:
                if train_args['fp16']:
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['max_grad_norm'])                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                global_step += 1

                # Log metrics
                if train_args['local_rank'] in [-1, 0] and train_args['logging_steps'] > 0 and global_step % train_args['logging_steps'] == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    record = {}
                    if train_args['local_rank'] == -1 and train_args['evaluate_during_training']:
                        results = evaluate(train_args, cache_validation_file, eval_json, model, tokenizer)
                        for key, value in results.items():
                            record["eval_{}".format(key)] = value
                    record["lr"]=scheduler.get_last_lr()[0]
                    record["loss"] = (tr_loss - logging_loss) / train_args['logging_steps']
                    logging_loss = tr_loss
                    wandb.log(record,step=global_step)
                    print(record)

                    replace_index = None
                    
                    if np.sum(BEST_F1 < (record['eval_f1'])) > 0:
                        if len(np.nonzero(BEST_F1 <  (record['eval_f1']))[0]) > 1:
                            replace_index = np.nonzero(BEST_F1 <  (record['eval_f1']))[0][0]
                        else:
                            replace_index = np.nonzero(BEST_F1 <  (record['eval_f1']))[0]
                            BEST_F1[replace_index] =  (record['eval_f1'])
                            BEST_STEP[replace_index] = global_step


                        # Save model checkpoint
                        if train_args['local_rank'] in [-1, 0] and train_args['save_steps'] > 0 and global_step % train_args['save_steps'] == 0 and replace_index is not None:
                            output_dir = os.path.join(train_args['output_dir'], "checkpoint-{}".format(global_step))
                            # Take care of distributed/parallel training
                            json_file = os.path.join(output_dir, "record.json")
                            config_file = os.path.join(output_dir, "model_config.json")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                            train_args['device'] = None
                            json.dump(train_args, open(config_file, "w"), indent=4)
                            train_args['device'] = torch.device("cuda")
                            tokenizer.save_pretrained(output_dir)
                            
                            torch.save(train_args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            json.dump(record, open(json_file, "w"), indent=4)
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
