source tests/training/common.env
source configs/config.env
for test_lm in ${test_lms[@]}; do
    echo "Testing classification training with model $test_lm"
    train_launch train.py --training_kind clf --model_name $test_lm \
    --output_dir $storage_dir/models/tmp_clf_model \
    --train_file tmp_test_data/tmp_clf.csv --output_column label  \
    --run_name test-clf-$test_lm $common_line
    rm -rf $storage_dir/models/tmp_clf_model
done

