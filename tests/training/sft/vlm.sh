source tests/training/common.env
source configs/config.env

tmp_dir="tmp_test_data/"
train_files=(tmp_vlm_single_train.csv tmp_vlm_multi_train.csv tmp_vlm_single_train.jsonl tmp_vlm_multi_train.jsonl)

for test_vlm in ${test_vlms[@]}; do
    for train_file in "${train_files[@]}"; do
        echo "Testing sft training with model $test_vlm on train file $train_file"
        python train.py --training_kind sft --modality vlm --model_name $test_vlm \
        --output_dir $storage_dir/models/tmp_vlm_sft_model \
        --train_file $tmp_dir/$train_file  \
        --run_name test-sft-$test_vlm-$train_file $common_line
        rm -rf $storage_dir/models/tmp_vlm_sft_model
    done
done

