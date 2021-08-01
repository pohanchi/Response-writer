echo "Setup enviroment" 
bash short_scripts/init.sh
bash short_scripts/download_data.sh

echo "extract coqa feature"
bash short_scripts/extract_coqa_feature.sh

echo "extract doqa feature"
bash short_scripts/extract_doqa_feature.sh

echo "extract quac feature"
bash short_scripts/extract_quac_feature.sh

echo "run doqa experiments"
bash short_scripts/doqa.sh

echo "run quac experiments"
bash short_scripts/quac.sh

echo "run coqa experiments"
bash short_scripts/coqa.sh