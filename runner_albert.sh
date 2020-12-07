
#train
python main_albert_doqa.py --config albert-official-config-train/DoQA/train_config_hf_albert_base-v2.yaml


# evaluate
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_hf_albert_base-v2_cooking.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_hf_albert_base-v2_movies.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_hf_albert_base-v2_travel.yaml



python main_albert_doqa.py --config albert-official-config-train/DoQA/train_config_hf_albert_large-v2.yaml


python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_hf_albert_large-v2_cooking.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_hf_albert_large-v2_movies.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_hf_albert_large-v2_travel.yaml


python main_albert_doqa.py --config albert-official-config-train/DoQA/train_config_original_albert_base-v2.yaml

python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_original_albert_base-v2_cooking.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_original_albert_base-v2_movies.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_original_albert_base-v2_travel.yaml


python main_albert_doqa.py --config albert-official-config-train/DoQA/train_config_original_albert_large-v2.yaml

python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_original_albert_large-v2_cooking.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_original_albert_large-v2_movies.yaml
python eval_albert_doqa.py --config albert-official-config-eval/DoQA/eval_config_original_albert_large-v2_travel.yaml
