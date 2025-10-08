source tests/training/common.env
source configs/config.env
for test_vlm in ${test_vlms[@]}; do
    echo "Testing sft training with model $test_vlm"
    train_launch train.py --training_kind sft --modality vlm --model_name $test_vlm \
    --output_dir $storage_dir/models/tmp_sft_model \
    --train_file tmp_test_data/tmp_vlm_po.csv --output_column label  \
    --run_name test-sft-$test_vlm $common_line
    rm -rf $storage_dir/models/tmp_sft_model
done

