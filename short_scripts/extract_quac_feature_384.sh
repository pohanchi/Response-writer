#!/user/bin/shell
python extract_feature_bert_quac_truncated.py --config preprocessing_configs/bert/QuAC/seq-384-128/base_dev_truncated_HAE.yaml
python extract_feature_bert_quac_truncated.py --config preprocessing_configs/bert/QuAC/seq-384-128/base_train_truncated_HAE.yaml