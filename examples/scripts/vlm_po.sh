source configs/config.env
accelerate launch train.py --training_kind dpo --model_name Qwen/Qwen2.5-VL-7B-Instruct --modality vlm \
--output_dir $storage_dir/models/vlm_po_model \
--train_file $storage_dir/data/manymodalqa/hf_po_train.csv --validation_file $storage_dir/data/manymodalqa/hf_po_val.csv \
--num_train_epochs 50 \
--per_device_train_batch_size 24 --per_device_eval_batch_size 24 \
--learning_rate 1e-4 --weight_decay 0.1 \
--logging_strategy steps --logging_steps 200 \
--eval_strategy steps --eval_steps 200 \
--save_strategy steps --save_steps 200 \
--early_stopping_patience 5 \
--load_best_model_at_end True \
--run_name manymodal-sft
