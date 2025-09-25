source tests/inference/common.env
for test_lm in $test_lms; do
    echo "Testing model: $test_lm"
    echo "Running inference with vLLM backend:"
    python infer.py --model_name $test_lm --input_file tmp_test_data/tmp_inference.csv --max_new_tokens 2 --ignore_checkpoint vllm
done