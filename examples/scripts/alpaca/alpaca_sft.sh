source configs/config.env
accelerate launch train.py --training_kind sft --model_name meta-llama/Meta-Llama-3-8B \
--output_dir $storage_dir/models/it_model \
--train_file $storage_dir/data/alpaca/train.csv \
--train_validation_split 0.9 \
--num_train_epochs 50 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--learning_rate 1e-4 \
--logging_strategy steps --logging_steps 200 \
--eval_strategy epoch --eval_steps 0.5 \
--lora_target_modules k_proj, v_proj, o_proj \
--save_strategy epoch --save_steps 0.5 \
--early_stopping_patience 5 \
--load_best_model_at_end True \
--run_name alpaca-sft

python infer.py --input_file $storage_dir/data/alpaca/test.csv \
--model_name $storage_dir/models/it_model/final_checkpoint \
--output_column model_output --max_new_tokens 200 \
--ignore_checkpoint hf --batch_size 12 --padding_side left

python infer.py --input_file $storage_dir/data/alpaca/test.csv --output_file $storage_dir/data/alpaca/test_og_output.jsonl \
--model_name meta-llama/Meta-Llama-3-8B \
--output_column model_output --max_new_tokens 200 \
--ignore_checkpoint hf --batch_size 12 --padding_side left

python << EOF
import pandas as pd
df = pd.read_json("$storage_dir/data/alpaca/test_og_output.jsonl", lines=True)
df2 = pd.read_json("$storage_dir/data/alpaca/test_output.jsonl", lines=True)
df2['og_output'] = df['model_output']
df2['show_og'] = df2['input'] + "\nModel Output: " + df2['og_output'].apply(lambda x: x[0] if isinstance(x, list) else x)
df2['show_it'] = df2['input'] + "\nModel Output: " + df2['model_output'].apply(lambda x: x[0] if isinstance(x, list) else x)
df2.to_csv("$storage_dir/data/alpaca/compare_outputs.csv", index=False)
EOF