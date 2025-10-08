source tests/training/common.env
source configs/config.env
for test_lm in ${test_lms[@]}; do
    echo "Testing dpo training with model $test_lm"
    train_launch train.py --training_kind dpo --model_name $test_lm \
    --output_dir $storage_dir/models/tmp_dpo_model \
    --train_file tmp_test_data/tmp_po.csv \
    --run_name test-dpo-$test_lm $common_line
    rm -rf $storage_dir/models/tmp_dpo_model
done