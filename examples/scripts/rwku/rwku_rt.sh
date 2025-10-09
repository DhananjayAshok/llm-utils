source configs/config.env
accelerate launch train.py --training_kind sft --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $storage_dir/models/rt_model \
--train_file $storage_dir/data/rwku/dpo_train.csv \
--train_validation_split 0.1 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
--learning_rate 1e-4 \
--logging_strategy epoch --logging_steps 0.1 \
--eval_strategy epoch --eval_steps 0.5 \
--lora_target_modules k_proj, v_proj, o_proj \
--save_strategy epoch --save_steps 0.5 \
--run_name rwku-rt

python infer.py --input_file $storage_dir/data/rwku/test.csv --output_file $storage_dir/data/rwku/test_rt_output.jsonl \
--model_name $storage_dir/models/rt_model/final_checkpoint \
--output_column model_output --max_new_tokens 200 \
--ignore_checkpoint hf --batch_size 4 --padding_side left