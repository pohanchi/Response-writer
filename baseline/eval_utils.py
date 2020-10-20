import torch
import os
import logging
import timeit
from preprocess_rc import *
from metrics.RC_metrics import *
from utils import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def evaluate(train_args, eval_file, model, tokenizer, prefix=""):
    preprocess= torch.load(eval_file)
    features, dataset, examples = preprocess['features'], preprocess['dataset'], preprocess['examples']

    if not os.path.exists(train_args['output_dir']) and train_args['local_rank'] in [-1, 0]:
        os.makedirs(train_args['output_dir'])

    train_args['eval_batch_size'] = train_args['batch_size']
    # Note that DistributedSampler samples randomly
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
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = RCResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
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

    predictions = compute_predictions_logits(
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
    )

    # Compute the F1 and exact scores.
    results = RC_evaluate(examples, predictions)
    return results
