mkdir ../dataset_local/CoQA
mkdir ../dataset_local/DoQA
mkdir ../dataset_local/QuAC

echo "Dowload CoQA train/dev json file to ../dataset_local/CoQA/{coqa-train-v1.0.json, coqa-dev-v1.0.json}"
wget -O ../dataset_local/CoQA/coqa-train-v1.0.json http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json
wget -O ../dataset_local/CoQA/coqa-dev-v1.0.json http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json

echo "Dowload QuAC train/val json file to ../dataset_local/QuAC/{train_v0.2.json, val_v0.2.json}"
wget -O ../dataset_local/QuAC/train_v0.2.json https://s3.amazonaws.com/my89public/quac/train_v0.2.json
wget -O ../dataset_local/QuAC/val_v0.2.json https://s3.amazonaws.com/my89public/quac/val_v0.2.json


wget http://ixa2.si.ehu.es/convai/doqa-v2.1.zip
unzip doqa-v2.1.zip
rm doqa-v2.1.zip
mv doqa-v2.1/doqa_dataset ../dataset_local/DoQA/
rm -rf doqa-v2.1

mkdir ../preprocessing_files/bert
mkdir ../preprocessing_files/bert/CoQA
mkdir ../preprocessing_files/bert/QuAC
mkdir ../preprocessing_files/bert/DoQA



