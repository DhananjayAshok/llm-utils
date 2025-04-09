storage_dir="" # set this to the storage directory in configs/private_vars.yaml
if [ -z "$storage_dir" ]; then
  echo "Please set the storage_dir variable in the script."
  exit 1
fi
accelerate launch train.py --output_dir ./tmp/sft --training_kind sft --train_file $storage_dir/data/sft/train.csv