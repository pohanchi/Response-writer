## __Hypernetwork for History Function__

Here is the implementation of the Hypernetwork for History Function. We utilize hypernetwork to generate parameter for storing conversation imformation. Then, through elementwise multiplication, parameter will fuse information into input to integrate conversation information and do the conversational questoin answering task. Also, we implement another four baseline methods to do a fair comparison in three conversational question answering datasets.

<p align="center">
  <img width=30% height=30% src="./model.png">
</p>

## __Structure__

The project structure need to be initialized like below 

```
|--main_project
    |--Model_storage
        |--Model 1 you have saved
        |--Model 2 you have saved
        |-- ...
    |--Dataset
        |--CoQA
        |--QuAC
        |--DoQA
        |--dataset_you_want
        |-- ...
    |--Preprocessing_files
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
