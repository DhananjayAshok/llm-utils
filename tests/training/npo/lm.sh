source tests/training/common.env
source configs/config.env
for test_lm in ${test_lms[@]}; do
    echo "Testing npo training with model $test_lm"
    train_launch train.py --training_kind npo --model_name $test_lm \
    --output_dir $storage_dir/models/tmp_npo_model \
    --train_file tmp_test_data/tmp_rwku_po.csv \
    --run_name test-npo-$test_lm $common_line
    rm -rf $storage_dir/models/tmp_npo_model
done