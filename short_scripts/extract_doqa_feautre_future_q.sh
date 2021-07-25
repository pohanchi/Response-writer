#!/user/bin/shell
python extract_feature_bert_doqa_question.py --config preprocessing_configs/bert/DoQA/question/bert_dev_doqa_cooking_HAE.yaml
python extract_feature_bert_doqa_question.py --config preprocessing_configs/bert/DoQA/question/bert_test_doqa_cooking_HAE.yaml
python extract_feature_bert_doqa_question.py --config preprocessing_configs/bert/DoQA/question/bert_test_doqa_movies_HAE.yaml
python extract_feature_bert_doqa_question.py --config preprocessing_configs/bert/DoQA/question/bert_test_doqa_travel_HAE.yaml
python extract_feature_bert_doqa_question.py --config preprocessing_configs/bert/DoQA/question/bert_train_doqa_cooking_HAE.yaml
