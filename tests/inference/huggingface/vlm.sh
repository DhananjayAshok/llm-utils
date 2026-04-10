source tests/inference/common.env
tmp_dir="tmp_test_data/"
input_files=(tmp_vlm_single_inference.csv tmp_vlm_single_inference.jsonl tmp_vlm_multi_inference.csv tmp_vlm_multi_inference.jsonl)

for test_vlm in $test_vlms; do
  for input_file in "${input_files[@]}"; do
    echo "Testing model: $test_vlm on input file: $input_file"
    python infer.py --modality vlm --model_name $test_vlm --input_file $tmp_dir/$input_file \
    --max_new_tokens 20 --ignore_checkpoint --dtype bfloat16 hf --batch_size 2 --padding_side left
  done
done