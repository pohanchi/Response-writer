#!/user/bin/shell

#History Function
echo "HHF training"
CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hf_bert_base.yaml

echo "HHF testing"
# # testing
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_base_cooking.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_base_movies.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hf_bert_base_travel.yaml

# HAE
echo "HAE training"
CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hf_bert_base_HAE.yaml

echo "HAE testing"
# testing
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_bert_base_cooking_HAE.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_bert_base_movies_HAE.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_bert_base_travel_HAE.yaml

echo "HisBERT training"
# HisBERT
CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hisbert.yaml

echo "HisBERT testing"
# testing
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hisbert_base_cooking.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hisbert_base_movies.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_hisbert_base_travel.yaml

echo "BERTQA_prepend training"
# original
CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_original_bert_base.yaml

echo "BERTQA_prepend testing"
# testing
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_baseline_bert_base_cooking.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_baseline_bert_base_movies.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_baseline_bert_base_travel.yaml


echo "BERTQA_original training"
# original_noq
CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_original_bert_base_noq.yaml


echo "BERTQA_original testing"
# testing
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_baseline_bert_base_cooking_noq.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_baseline_bert_base_movies_noq.yaml
CUDA_VISIBLE_DEVICES=0 python eval_bert_doqa.py --config bert-official-config-eval/DoQA/eval_config_baseline_bert_base_travel_noq.yaml

