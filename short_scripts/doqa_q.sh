#!/user/bin/shell

#History Function
CUDA_VISIBLE_DEVICES=1 python main_bert_doqa.py --config bert-official-config-train/DoQA/question/train_config_hf_bert_base.yaml

# testing
CUDA_VISIBLE_DEVICES=1 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/question/eval_config_hf_bert_base_cooking.yaml
CUDA_VISIBLE_DEVICES=1 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/question/eval_config_hf_bert_base_movies.yaml
CUDA_VISIBLE_DEVICES=1 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/question/eval_config_hf_bert_base_travel.yaml

