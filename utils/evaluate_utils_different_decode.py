import torch
import os
import logging
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,SubsetRandomSampler
from extract_feature.extract_feature_bert_doqa_different_decode import *
from metrics.RC_metrics_quac import *
from metrics.quac_metrics import *
from .utils import *
import IPython
import pdb


def evaluate(train_args, eval_file, eval_json, model, tokenizer, beam_search, prefix=""):
    
    preprocess= torch.load(eval_file)
    
    example_dict = preprocess['example_dict']

    if not os.path.exists(train_args['output_dir']) and train_args['local_rank'] in [-1, 0]:
        os.makedirs(train_args['output_dir'])

    train_args['eval_batch_size'] = train_args['batch_size']
    # Note that DistributedSampler samples randomly

    turn_prefix = "turn_"
    turn_len = len(list(example_dict.keys()))
    history_turns = {}
    all_examples = []
    for turn_ids in range(turn_len):
        key = turn_prefix + str(turn_ids)

        subset_dataset = example_dict[key]

        examples = convert_datalist_to_examples(subset_dataset, history_turns, beam_search)

        features, dataset = convert_examples_to_features(examples, tokenizer=tokenizer, is_training=False)

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=train_args['eval_batch_size'])

        # multi-gpu evaluate
        if train_args['n_gpu'] > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", train_args['eval_batch_size'])

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(train_args['device']) for t in batch)

            with torch.no_grad():
                
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
                    "start_positions":None,
                    "end_positions": None,
                    "history_starts":batch[12] if len(batch) >= 14 else None,
                    "history_ends": batch[13] if len(batch) >= 14 else None,
                }

                feature_indices = batch[11]

                # XLNet and XLM use more arguments for their predictions
                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):

                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = RCResult(unique_id, start_logits, end_logits)

                all_results.append(result)
        

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        output_prediction_file = os.path.join(train_args['output_dir'], "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(train_args['output_dir'], "nbest_predictions_{}.json".format(prefix))

        if train_args['version_2_with_negative']:
            output_null_log_odds_file = os.path.join(train_args['output_dir'], "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        predictions, official_all_predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            train_args['n_best_size'],
            train_args['max_answer_length'],
            train_args['do_lower_case'],
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            train_args['verbose_logging'],
            train_args['version_2_with_negative'],
            train_args['null_score_diff_threshold'],
            tokenizer,
            beam_search,
        )

        key_set = list(predictions.keys())
        for key in key_set:
            key_subset = list(predictions[key].keys())
            for key_s in key_subset:
                history_turns[key_s] = {}
                history_turns[key_s]['prediction'] = {}
                history_turns[key_s]['prediction']['text'] = predictions[key][key_s]['best_span_str']
                history_turns[key_s]['prediction']['answer_start'] = predictions[key][key_s]['answer_start']
                history_turns[key_s]['prediction']['answer_end'] = predictions[key][key_s]['answer_end']
                history_turns[key_s]['question_answer_string'] = predictions[key][key_s]['question_answer_string']
                history_turns[key_s]['history_span_path'] = predictions[key][key_s]['history_span_path']
        # import ipdb; ipdb.set_trace()
        
        all_examples.extend(examples)
    print('after aggregate all history turn, decode in the final')
    features, dataset = convert_examples_to_features(all_examples, tokenizer=tokenizer, is_training=False)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=train_args['eval_batch_size'])

    # multi-gpu evaluate
    if train_args['n_gpu'] > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", train_args['eval_batch_size'])

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(train_args['device']) for t in batch)

        with torch.no_grad():
            
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
                "start_positions":None,
                "end_positions": None,
                "history_starts":batch[12] if len(batch) >= 14 else None,
                "history_ends": batch[13] if len(batch) >= 14 else None,
            }

            feature_indices = batch[11]

            # XLNet and XLM use more arguments for their predictions
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):

            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = RCResult(unique_id, start_logits, end_logits)

            all_results.append(result)
    

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(train_args['output_dir'], "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(train_args['output_dir'], "nbest_predictions_{}.json".format(prefix))

    if train_args['version_2_with_negative']:
        output_null_log_odds_file = os.path.join(train_args['output_dir'], "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    predictions, official_all_predictions = compute_predictions_logits(
        all_examples,
        features,
        all_results,
        train_args['n_best_size'],
        train_args['max_answer_length'],
        train_args['do_lower_case'],
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        train_args['verbose_logging'],
        train_args['version_2_with_negative'],
        train_args['null_score_diff_threshold'],
        tokenizer,
        beam_search,
    )



    val = json.load(open(eval_json, 'r'))['data']
    
    metric_json = eval_fn(val, official_all_predictions, False)

    output_metrics_file = os.path.join(train_args['output_dir'], "metrics.json".format(prefix))

    with open(output_metrics_file, 'w') as fout:
      json.dump(metric_json, fout)

    # Compute the F1 and exact scores.
    return metric_json