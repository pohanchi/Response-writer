#!/user/bin/shell
pip install spacy && python -m spacy download en

echo "Start to extract CoQA dev set feature"

python extract_feature/extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/base_dev_truncated_HAE.yaml

echo "finish extract CoQA dev set feature"

echo "Start to extract CoQA training set feature"

python extract_feature/extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/base_train_truncated_HAE.yaml

echo "finish extract CoQA training set feature"
# python extract_feature/extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/base_dev_truncated_HAE_6_turn.yaml
# python extract_feature/extract_feature_bert_coqa_truncated.py --config preprocessing_configs/bert/CoQA/base_train_truncated_HAE_6_turn.yaml

