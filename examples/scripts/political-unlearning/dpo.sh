source configs/config.env
python train.py --training_kind dpo --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $storage_dir/models/political_dem_model \
--train_file $storage_dir/data/political-unlearning/democrat.csv \
--train_validation_split 0.9 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
--learning_rate 1e-4 \
--logging_strategy epoch --logging_steps 0.1 \
--eval_strategy epoch --eval_steps 0.5 \
--load_best_model_at_end True \
--early_stopping_patience 2 \
--lora_target_modules k_proj, v_proj, o_proj \
--save_strategy epoch --save_steps 0.5 \
--run_name political-dem-dpo \
--push_to_hub True --hub_model_id unlearn_democrats_Llama3_8b

python train.py --training_kind dpo --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $storage_dir/models/political_rep_model \
--train_file $storage_dir/data/political-unlearning/republican.csv \
--train_validation_split 0.9 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
--learning_rate 1e-4 \
--logging_strategy epoch --logging_steps 0.1 \
--eval_strategy epoch --eval_steps 0.5 \
--load_best_model_at_end True \
--early_stopping_patience 2 \
--lora_target_modules k_proj, v_proj, o_proj \
--save_strategy epoch --save_steps 0.5 \
--run_name political-rep-dpo \
--push_to_hub True --hub_model_id unlearn_republicans_Llama3_8b
