source tests/inference/common.env
for test_vlm in $test_vlms; do
    echo "Testing model: $test_vlm"
  python infer.py --modality vlm --model_name $test_vlm --input_file tmp_test_data/tmp_vlm_inference.csv \
  --max_new_tokens 20 --ignore_checkpoint --dtype bfloat16 hf --batch_size 2 --padding_side left
done