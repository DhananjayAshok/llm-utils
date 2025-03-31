storage_dir="" # set this to the storage directory in configs/private_vars.yaml
accelerate launch train.py --output_dir ./tmp/sft --training_kind sft --train_file $storage_dir/data/sft/train.csv