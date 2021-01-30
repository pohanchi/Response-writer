
# training
python main_roberta_doqa.py --config roberta-official-config-train/DoQA/train_config_hf_roberta_base.yaml

# # evaluating
python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_hf_roberta_base_cooking.yaml
python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_hf_roberta_base_movies.yaml
python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_hf_roberta_base_travel.yaml


python main_roberta_doqa.py --config roberta-official-config-train/DoQA/train_config_hf_roberta_large.yaml


python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_hf_roberta_large_cooking.yaml
python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_hf_roberta_large_movies.yaml
python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_hf_roberta_large_travel.yaml


# python main_roberta_doqa.py --config roberta-official-config-train/DoQA/train_config_original_roberta_base.yaml


# python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_original_roberta_base_cooking.yaml
# python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_original_roberta_base_movies.yaml
# python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_original_roberta_base_travel.yaml



# python main_roberta_doqa.py --config roberta-official-config-train/DoQA/train_config_original_roberta_large.yaml



# python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_original_roberta_large_cooking.yaml
# python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_original_roberta_large_movies.yaml
# python eval_roberta_doqa.py --config roberta-official-config-eval/DoQA/eval_config_original_roberta_large_travel.yaml
