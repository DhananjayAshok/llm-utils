# Language Model Utilities

Useful code for training and inference of Language Models. I currently support the following functionality:

Language Models:
1. Inference (with HuggingFace Transformers and vLLM)
2. Pretraining
3. Finetuning (Classification and SFT for Generation)
4. Preference Optimization / RL Training

Vision Language Models:
1. Inference
2. Finetuning (Classification and SFT for Generation)
3. Preference Optimization / RL Training

All code is based on HuggingFace Transformers and TRL and supports multiple GPUs as well as quantization. 

## Setup

Follow the [instructions](setup/README.md) to set up the environment with the right packages and Python version. Then, before running anything you should make sure to populate the essential fields in the [config files](configs/README.md).



Log in to WandB with 

```bash
wandb login
```

Set up the accelerate config file with
```bash
accelerate config
```
Common Setup:
- This Machine
- multi-GPU 
- 1 node
- No checking distributed ops
- No torch Dynamo 
- Enter number of available GPUs when asked 
- mixed precision bf16

Basic Setup:
- No DeepSpeed, FSDP, Megatron
- yes numa efficiency

FSDP:
- No DeepSpeed
- Yes FSDP
- FSDP version 2
- Choose defaults for `enable resharding` (yes), `offload` (no)
- Transformer Based Wrap => yes to use the model's _no_split_modules
- SHARDED_STATE_DICT state dict type
- Yes to CPU RAM efficient model loading
- No to activation checkpointing
- No to parallelism config

The FSDP configuration gives me the accelerate config (at ....huggingface/accelerate/default_config.yaml) yaml:

```yaml
compute_environment: LOCAL_MACHINE                                                                                                             
debug: false                                                                                                                                   
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Examples

I have a [set of examples](examples/README.md) that show how to use the code for different tasks. The examples cover all functionality of the code. 

## Project Organization

### Inference

#### Language Model Inference

#### Vision Language Model Inference

### Training


#### LM Training

#### VLM Training

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