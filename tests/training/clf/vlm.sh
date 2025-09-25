source tests/training/common.env
source configs/config.env
for test_vlm in ${test_vlms[@]}; do
    echo "Testing classification training with model $test_vlm"
    accelerate launch train.py --training_kind clf --model_name $test_vlm \
    --output_dir $storage_dir/models/tmp_clf_model --max_steps 10  \
    --train_file tmp_test_data/tmp_vlm_clf.csv --train_validation_split 0.85 --validation_test_split 0.1 --output_column label  \
    --logging_strategy steps --eval_strategy steps --save_strategy steps --save_steps 5 --logging_steps 1 --eval_steps 5 --load_best_model_at_end \
    --run_name test-clf-$test_vlm --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    rm -rf $storage_dir/models/tmp_clf_model
done

