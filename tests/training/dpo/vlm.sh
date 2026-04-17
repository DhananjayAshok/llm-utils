source tests/training/common.env
source configs/config.env
for test_vlm in ${test_vlms[@]}; do
    echo "Testing dpo training with model $test_vlm"
    python train.py --training_kind dpo --modality vlm --model_name $test_vlm \
    --output_dir $storage_dir/models/tmp_vlm_dpo_model \
    --train_file tmp_test_data/tmp_vlm_po.csv  \
    --run_name test-dpo-$test_vlm $common_line
    rm -rf $storage_dir/models/tmp_vlm_dpo_model
done

