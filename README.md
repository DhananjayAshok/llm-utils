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

That's all the setup you need to do for inference, but for training you will need to set up the accelerate config file.

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

That's it! You can now run the code.

## Examples

I have a [set of examples](examples/README.md) that show how to use the code for different tasks. The examples cover all functionality of the code. 

## Project Organization

### Inference

The entry point for inference is the [`infer.py`](infer.py) script. It supports both HuggingFace Transformers and vLLM inference pipelines for both Language Models and Vision Language Models. The call to inference has three components:
1. Core arguments: These are found in the click declaration of the function [`main`](infer.py) and should be passed in right after the filename with `python infer.py --modality vlm` etc
2. Framework selection: This is done with the `hf` or `vllm` command, which selects the HuggingFace Transformers or vLLM inference pipeline respectively. e.g. `python infer.py --model_name <name> hf`
3. Framework specific arguments: These are passed in after the `hf` or `vllm` command. For example, `python infer.py --model_name <name> hf --batch_size 8` will run inference with the HuggingFace Transformers pipeline. See the [huggingface](infer/huggingface_inference.py) and [vllm](infer/vllm_inference.py) inference files for the arguments that can be passed in after the `hf` or `vllm` command.

The scripts will expect your input to be a csv file with a column named `input` that contains the text to be processed (and `image` with a url or path to an image for VLMs). It also expects that the input file does *not* contain the columns `output` or `inference_completed`. The output will be saved in the same directory as the input file, with a suffix `_output` added to the filename, and as a json lines file (`.jsonl`). The names of the columns can be changed with the appropriate arguments.  

This output file also automatically acts as a checkpoint if inference stops halfway, and unless you tell it not to, the code will always try to restart from a checkpoint. 

### Training

The entry point for training is the [`train.py`](train.py) script. It supports pretraining, supervised finetuning, classification finetuning and direct preference optimization (DPO) training. There are two sets of arguments this script accepts:
1. ScriptArguments: Check these out in the [`ScriptArguments`](train.py) class. 
2. Learning specific arguments: These depend on the kind of training you are doing, and all of them are taken from HuggingFace or HuggingFace TRL. Classification takes in the same arguments as [TrainingArguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py), Supervised Finetuning takes in the same arguments as [SFTConfig](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py) and Direct Preference Optimization takes in the same arguments as [DPOConfig](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py).

To see the parameters that can be used on the command line see the respective Config files. All arguments that are used internally in a Trainer Class (i.e. not arguments like --do_eval, but arguments like --save_every) are passed on to the Trainer class. So, for example, if you want to set the number of epochs for classification training, you must add `--num_train_epochs <number>` to the set of args passed in.

The essential format to follow for each training paradigm is given below:

1. Pretraining: input files can either be .txt or .csv, csv must have column input with the text to learn. If you want to do supervised finetuning but fit loss on the prompt as well, then this is handled by this training mode. In this case you must specify a csv file with input and output columns, and make sure the relevant ScriptArgument parameter is set to true
2. Supervised Finetuning: input files must be a csv with input and output columns. Loss is only computed on completions
3. Direct Preference Optimization: input files must be a csv with input, chosen and rejected columns. 

WandDB is used to log the metrics, and you can always recover the history of a prior run with:

```python
from utils import get_history
history = get_history("run_name")
```