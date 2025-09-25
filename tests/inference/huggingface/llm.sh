source tests/inference/common.env
for test_lm in $test_lms; do
    echo "Testing model: $test_lm"
    echo "Running Discover Batch Size Utility:"
    python infer.py --model_name $test_lm --input_file tmp_test_data/tmp_inference.csv --max_new_tokens 2 --num_return_sequences 2 --ignore_checkpoint infer_batch_size --num_beams 2

    echo "Running Inference with HF backend:"
    python infer.py --model_name $test_lm --input_file tmp_test_data/tmp_inference.csv --max_new_tokens 2 --ignore_checkpoint hf --batch_size 4 --padding_side left
done