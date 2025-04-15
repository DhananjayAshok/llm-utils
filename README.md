# Language Model Utilities

Useful code for training and inference of Language Models. 

## Examples

You can load in sample data with:

```bash
python create_examples.py generate
```

You can specify the specific variants you want with:

```bash
python create_examples.py generate --variants sft --variants pref
```

Available variants are: 
1. pre: Pretraining
2. sft: Supervised Finetuning (text)
3. clf: Classification 
4. pref: Preference Optimization


Before training, you have to install dependancies with:

```bash
scripts/create_env.sh
```

Log in to WandB with 

```bash
wandb login
```

Set up the accelerate config file. As a default I use multi-GPU FSDP with Torch Dynamo (inductor) speed up (no quantization). All training scripts here are LoRA, and 

To set this up you can do

```bash
accelerate config
```

When going through the options, select the options that correspond to:

```yaml
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_reshard_after_forward: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false # must be set to true if you want to use torch dynamo
  fsdp_version: 1
mixed_precision: bf16
```



Then, run the script you want to train with using:



```bash
python examples/scripts/sft.sh
```

To see the parameters that can be used on the command line (such as `--per_device_train_batch_size', --eval_strategy, --max_steps or --num_train_epochs, --max_length) see the respective Config files for 

1. [SFTConfig](github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py) 
2. [DPOConfig](github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py)
3. [TrainingArguments](github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)

## Project organization

The inference code: 

The training code is essentially a quick wrapper around HuggingFace Trainer and TRL. In general the only required arguments are training_kind, model_name and a train_file for data. The call looks like

```bash
python train.py --training_kind <pre/sft/clf/dpo> --model_name <name> --output_dir tmp/ --training_file <something>
```




The essential format to follow for each training paradigm is given below:

1. Pretraining: input files can either be .txt or .csv, csv must have column input with the text to learn. If you want to do supervised finetuning but fit loss on the prompt as well, then this is handled by this training mode. In this case you must specify a csv file with input and output columns, and make sure the relevant ScriptArgument parameter is set to true
2. Supervised Finetuning: input files must be a csv with input and output columns. Loss is only computed on completions
3. Direct Preference Optimization: input files must be a csv with input, chosen and rejected columns. 