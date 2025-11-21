#model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name="mistralai/Mistral-7B-Instruct-v0.3"
model_short_name="${model_name#*/}"
source configs/config.env
rm -rf $storage_dir/models/political_dem_model
rm -rf $storage_dir/models/political_rep_model
python train.py --training_kind dpo --model_name $model_name \
--output_dir $storage_dir/models/political_dem_model \
--train_file $storage_dir/data/political_unlearning/democrat.csv \
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
--run_name political-dem-dpo-$model_short_name \
--push_to_hub True --hub_model_id unlearn_democrats_$model_short_name

python train.py --training_kind dpo --model_name $model_name \
--output_dir $storage_dir/models/political_rep_model \
--train_file $storage_dir/data/political_unlearning/republican.csv \
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
--run_name political-rep-dpo-$model_short_name \
--push_to_hub True --hub_model_id unlearn_republicans_$model_short_name
