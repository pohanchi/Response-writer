#!/user/bin/shell
python extract_feature_bert_doqa_future.py --config preprocessing_configs/bert/DoQA/future/future_eval/bert_dev_doqa_cooking_HAE.yaml --future
python extract_feature_bert_doqa_future.py --config preprocessing_configs/bert/DoQA/future/future_eval/bert_test_doqa_cooking_HAE.yaml --future
python extract_feature_bert_doqa_future.py --config preprocessing_configs/bert/DoQA/future/future_eval/bert_test_doqa_movies_HAE.yaml --future
python extract_feature_bert_doqa_future.py --config preprocessing_configs/bert/DoQA/future/future_eval/bert_test_doqa_travel_HAE.yaml --future
