#!/user/bin/shell

echo "start extract QuAC dev set feature"
python extract_feature/extract_feature_bert_quac_truncated.py --config preprocessing_configs/bert/QuAC/base_dev_truncated_HAE.yaml
echo "finish extract QuAC training set feature"

echo "start extract QuAC training set feature"
python extract_feature/extract_feature_bert_quac_truncated.py --config preprocessing_configs/bert/QuAC/base_train_truncated_HAE.yaml
echo "finish extract QuAC training set feature"
