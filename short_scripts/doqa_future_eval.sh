#!/user/bin/shell

# #History Function
# CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/future/train_config_hf_bert_base.yaml

# testing
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/future/future_eval/eval_config_hf_bert_base_cooking.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/future/future_eval/eval_config_hf_bert_base_movies.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/future/future_eval/eval_config_hf_bert_base_travel.yaml

