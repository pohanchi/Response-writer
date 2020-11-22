# build pipeline

# create data
python extract_feature_bert_doqa.py --config create_feature/bert_dev_doqa_cooking.yaml
python extract_feature_bert_doqa.py --config create_feature/bert_test_doqa_cooking.yaml
python extract_feature_bert_doqa.py --config create_feature/bert_train_doqa_cooking.yaml
python extract_feature_bert_doqa.py --config create_feature/bert_test_doqa_movies.yaml
python extract_feature_bert_doqa.py --config create_feature/bert_dev_doqa_travel.yaml

mkdir ../CQAModel/memory

# train bert config
python main_bert_doqa.py --config train_bert_config/train_config_bert_sep_memory.yaml

# eval bert config
python eval_bert_doqa.py --config eval_bert_config/eval_config_bert_cooking_sep_memory.yaml
python eval_bert_doqa.py --config eval_bert_config/eval_config_bert_movies_sep_memory.yaml
python eval_bert_doqa.py --config eval_bert_config/eval_config_bert_travel_sep_memory.yaml