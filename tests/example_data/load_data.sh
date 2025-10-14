echo "Testing basic data set up code"
python configs/create_env_file.py
python create_examples.py setup --dataset_names pubmedqa --dataset_names manymodalqa --dataset_names rwku --dataset_names alpaca