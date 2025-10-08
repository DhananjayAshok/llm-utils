source tests/training/common.env
source configs/config.env
for test_vlm in ${test_vlms[@]}; do
    echo "Testing classification training with model $test_vlm"
    train_launch train.py --training_kind clf --modality vlm --model_name $test_vlm \
    --output_dir $storage_dir/models/tmp_clf_model \
    --train_file tmp_test_data/tmp_vlm_clf.csv --output_column label  \
    --run_name test-clf-$test_vlm $common_line
    rm -rf $storage_dir/models/tmp_clf_model
done

