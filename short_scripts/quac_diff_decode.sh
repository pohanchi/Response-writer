#!/user/bin/shell


CUDA_VISIBLE_DEVICES=0 python different_decoding_quac.py --config bert-official-config-eval/QuAC/diff_decoding/eval_config_bert_base_HAE.yaml
CUDA_VISIBLE_DEVICES=0 python different_decoding_quac.py --config bert-official-config-eval/QuAC/diff_decoding/eval_config_baseline_bert_base.yaml
CUDA_VISIBLE_DEVICES=0 python different_decoding_quac.py --config bert-official-config-eval/QuAC/diff_decoding/eval_config_hf_bert_base23.yaml
CUDA_VISIBLE_DEVICES=0 python different_decoding_quac.py --config bert-official-config-eval/QuAC/diff_decoding/eval_config_hisbert_base.yaml
