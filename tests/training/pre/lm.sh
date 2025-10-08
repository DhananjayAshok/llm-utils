source tests/training/common.env
source configs/config.env
for test_lm in ${test_lms[@]}; do
    echo "Testing pre training with model $test_lm"
    train_launch train.py --training_kind pre --model_name $test_lm \
    --output_dir $storage_dir/models/tmp_pre_model \
    --train_file tmp_test_data/tmp_ft.csv \
    --run_name test-pre-$test_lm $common_line
    rm -rf $storage_dir/models/tmp_pre_model
done