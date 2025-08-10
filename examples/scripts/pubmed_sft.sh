source examples/scripts/env.sh
accelerate launch train.py --training_kind sft --model_name meta-llama/Llama-3.2-1B-Instruct \
--output_dir $storage_dir/models/ft_model \
--num_train_epochs 150 --train_file $storage_dir/data/pubmedqa/hf_ft_train.csv \
--per_device_train_batch_size 24 --per_device_eval_batch_size 24 \
--learning_rate 2e-4 --weight_decay 0.01 \
--train_validation_split 0.9  --logging_strategy epoch --eval_strategy epoch --save_strategy epoch --load_best_model_at_end True \
--run_name pubmed-sft
