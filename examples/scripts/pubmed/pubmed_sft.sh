source configs/config.env
accelerate launch train.py --training_kind sft --model_name meta-llama/Llama-3.2-1B-Instruct \
--output_dir $storage_dir/models/ft_model \
--train_file $storage_dir/data/pubmedqa/hf_ft_train.csv --validation_file $storage_dir/data/pubmedqa/hf_ft_val.csv \
--num_train_epochs 50 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--learning_rate 1e-4 \
--logging_strategy steps --logging_steps 200 \
--eval_strategy epoch --eval_steps 0.5 \
--lora_target_modules k_proj, v_proj, o_proj \
--save_strategy epoch --save_steps 0.5 \
--early_stopping_patience 5 \
--load_best_model_at_end True \
--run_name pubmed-sft

python infer.py --input_file $storage_dir/data/pubmedqa/hf_ft_val.csv \
--model_name $storage_dir/models/ft_model/final_checkpoint \
--output_column model_output --max_new_tokens 200 \
--ignore_checkpoint hf --batch_size 12 --padding_side left