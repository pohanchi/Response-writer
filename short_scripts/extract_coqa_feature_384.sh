#!/user/bin/shell
python extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/seq-384-128/base_dev_truncated_HAE.yaml
python extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/seq-384-128/base_train_truncated_HAE.yaml

# python extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/base_dev_truncated_HAE_6_turn.yaml
# python extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/base_train_truncated_HAE_6_turn.yaml

