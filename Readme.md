## __Hypernetwork for History Function__

Here is the implementation of the Hypernetwork for History Function. We utilize hypernetwork to generate parameter for storing conversation imformation. Then, through elementwise multiplication, parameter will fuse information into input to integrate conversation information and do the conversational questoin answering task. Also, we implement another four baseline methods to do a fair comparison in three conversational question answering datasets.

<p align="center">
  <img width=30% height=30% src="./model.png">
</p>

## Dependencies

- Python 3.7.3
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.
- Required packages and their use are listed [requirements.txt](requirements.txt).
- `pip install -r requirements.txt`
- `apex` for half precision training (Please manually install in your computer)

## Quick Start
You can do `bash short_scripts/total_pipeline.sh` to run all pipeline ( init strucuture -> download data -> preprocessing -> train model ) after you well installed all package and finished the dependencies stage (apex you need to manually install).

## __Structure__

The project structure need to be initialized like below.

You can directly run the short script `bash ./short_script/init.sh` to construct the structure (one time is enough).

```
|--main_project
    |--Model_storage
        |--Model 1 you have saved
        |--Model 2 you have saved
        |-- ...
    |--dataset_local
        |--CoQA
            |--coqa-dev-v1.0.json
            |--coqa-train-v1.0.json
            |-- ...
        |--QuAC
            |--train.json
            |--dev.json
            |-- ...
        |--DoQA
        |--dataset_you_want
        |-- ...
    |--preprocessing_files
        |--CoQA
        |--DoQA
        |--QuAC
        |--dataset_you_want
        |-- ...
    |--Response-writer
        |--bert-official-config-eval
            |-- ...
            ...
        |--bert-official-config-train
            |-- ...
            ...
        |-- ...
        ...
        different_decoding_coqa.py
        different_decoding_doqa.py
        different_decoding_quac.py
        ...
        model.png
        Readme.md
        requirement.txt
        ...
```

In the following tutorial, we will show you how to run this code on DoQA/CoQA/QuAC dataset.

## __Stage 0: Prepare data__

In Stage 0, you need to download json file of dataset and store in the dataset_local folder. (For example: dataset_local/QuAC/train.json and dataset_local/QuAC/dev.json).
You can install manually by youself. Here, we provide a conviencent shell script to automatically download CoQA, QuAC, dataset.

`bash short_scripts/download_data.sh`

## __Stage 1: Data Preprocessing__

In Stage 1, We have had json file, so we need to do preprocessing and store feature to do conversational question answering.
You can directly run the well-written short script to get CoQA, DoQA, QuAC dataset preprocessing.

`bash short_script/extract_coqa_feature.sh`

`bash short_script/extract_doqa_feature.sh`

`bash short_script/extract_quac_feature.sh`

## __Stage 2: Train & Evaluate your Model__

In Stage 2, You can train and evaluate your model.

`bash short_scripts/doqa.sh`

`bash short_scripts/coqa.sh`

`bash short_scripts/quac.sh`

## __Model Checkpoint__

The google drive link is [here](https://drive.google.com/drive/folders/1ev-gOmNL6z2kX3cwTtiuIgtzSOKrmZZw?usp=sharing).

Below are the description of checkpoint name:

* Bert-base: PHQA (Prepend History question, Answer)
* HisBERT: HisBERT
* BERT_HAE: History Answer Encoding
* BERT_HHF: Hypernetwork for History Function
 
