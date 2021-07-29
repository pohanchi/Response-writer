# !/user/bin/shell
python extract_feature/extract_feature_bert_doqa.py --config preprocessing_configs/bert/DoQA/bert_dev_doqa_cooking_HAE.yaml
python extract_feature/extract_feature_bert_doqa.py --config preprocessing_configs/bert/DoQA/bert_test_doqa_cooking_HAE.yaml
python extract_feature/extract_feature_bert_doqa.py --config preprocessing_configs/bert/DoQA/bert_test_doqa_movies_HAE.yaml
python extract_feature/extract_feature_bert_doqa.py --config preprocessing_configs/bert/DoQA/bert_test_doqa_travel_HAE.yaml
python extract_feature/extract_feature_bert_doqa.py --config preprocessing_configs/bert/DoQA/bert_train_doqa_cooking_HAE.yaml
