#!/user/bin/shell


# # HAE
# CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hf_bert_base_HAE.yaml

# testing
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_bert_base_cooking_HAE.yaml
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_bert_base_movies_HAE.yaml
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_bert_base_travel_HAE.yaml


#History Function
# CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hf_bert_base.yaml

# testing
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hf_bert_base_cooking.yaml
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hf_bert_base_movies.yaml
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hf_bert_base_travel.yaml

# # HisBERT
# CUDA_VISIBLE_DEVICES=0 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_hisbert.yaml

# # testing
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hisbert_base_cooking.yaml
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hisbert_base_movies.yaml
# CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hisbert_base_travel.yaml

# original
# CUDA_VISIBLE_DEVICES=1 python main_bert_doqa.py --config bert-official-config-train/DoQA/train_config_original_bert_base.yaml

# testing
CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hf_bert_base_cooking.yaml
CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hf_bert_base_movies.yaml
CUDA_VISIBLE_DEVICES=0 python different_decoding_doqa.py --config bert-official-config-eval/DoQA/diff_decoding/eval_config_hf_bert_base_travel.yaml