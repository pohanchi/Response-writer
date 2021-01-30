
#training
python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hf_bert_base.yaml

# evaluating
python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_base_cooking.yaml
python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_base_movies.yaml
python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_base_travel.yaml

python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hf_bert_large.yaml


python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_large_cooking.yaml
python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_large_movies.yaml
python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_large_travel.yaml


# python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_original_bert_base.yaml

# python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_original_bert_base_cooking.yaml
# python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_original_bert_base_movies.yaml
# python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_original_bert_base_travel.yaml


# python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_original_bert_large.yaml


# python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_original_bert_large_cooking.yaml
# python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_original_bert_large_movies.yaml
# python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_original_bert_large_travel.yaml
