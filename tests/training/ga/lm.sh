source tests/training/common.env
source configs/config.env
for test_lm in ${test_lms[@]}; do
    echo "Testing ga training with model $test_lm"
    train_launch train.py --training_kind ga --model_name $test_lm \
    --output_dir $storage_dir/models/tmp_ga_model \
    --train_file tmp_test_data/tmp_rwku_ft.csv \
    --run_name test-ga-$test_lm $common_line
    rm -rf $storage_dir/models/tmp_ga_model
done