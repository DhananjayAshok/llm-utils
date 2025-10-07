source configs/config.env
batch_size=10
file_names="color shape"
for file_name in $file_names; do
  python infer.py --modality vlm --model_name llava-hf/llava-v1.6-mistral-7b-hf --input_file $storage_dir/data/manymodalqa/${file_name}.csv \
  --max_new_tokens 200 --ignore_checkpoint --dtype bfloat16 hf --batch_size $batch_size --padding_side left
done