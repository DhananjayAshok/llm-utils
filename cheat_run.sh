export LEARNING_RATE=5e-5
export BATCH_SIZE=2
export EPOCH=2
export CKPT_PATH=meta-llama/Meta-Llama-3.1-8B
export SAVE_NAME=cheatmodel
export SAVE_PATH=${SAVE_NAME}

#    --report_to="wandb" \
python cheat_script.py \
    --model_name_or_path=${CKPT_PATH} \
    --learning_rate=${LEARNING_RATE} \
    --auto_find_batch_size \
    --gradient_accumulation_steps=16 \
    --fsdp="full_shard auto_wrap offload" \
    --fsdp_config="fsdp.json" \
    --gradient_checkpointing \
    --dataset_text_field="text" \
    --max_seq_length=200 \
    --output_dir=${SAVE_PATH} \
    --logging_steps=1 \
    --num_train_epochs=${EPOCH} \
    --max_steps=-1 \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj"\
    --torch_dtype=bfloat16 \
    --bf16=True \
    2>&1 | tee ${SAVE_NAME}.log