# Language Model Utilities

Useful code for training and inference of Language Models. I currently support the following functionality:

Language Models:
1. Inference with HuggingFace Transformers and vLLM (no vLLM support for VLMs at the moment)
2. Pretraining
3. Finetuning (Classification and Supervised Finetuning for Generation)
4. Preference Optimization (Direct Preference Optimization, Contrastive Preference Optimization)
5. Unlearning (Gradient Ascent, Negative Preference Optimization)

All code is based on HuggingFace Transformers and TRL and supports FSDP with multiple GPUs. 

This branch is being actively worked on and may have breaking changes pushed to it at any time. If you want to use a stable version of the code base without running the tests / actively adding features, see the [app]() branch instead. 

## Setup
First, clone the repo, then follow the [instructions](setup/README.md) to set up the environment with the right packages and Python version. Before running anything, you should make sure to populate the essential fields in the [config files](configs/README.md). After that, run:

```bash
python configs/create_env_file.py
```

That's all the setup you need for inference, but for training, you will need to set up a couple of additional things. 

Log in to WandB with 

```bash
wandb login
```

That's it! You can now run the code. Test that the code base works fine by running:
```bash
bash tests/test_all.sh
```

If this fails, then isolate the problem by following the [test instructions](tests/README.md)

### FSDP (Optional)
If you want to use FSDP or Accelerate distribution, then set up the accelerate config file. In general, I recommend you do *not* do this unless you know what you're doing / you have a good reason to try it. Accelerate FSDP etc. is a bit buggy and can considerably slow down model training / saving for what seems to be little gain. 

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

The FSDP configuration gives me the accelerate config (at `<path_to_huggingface>/accelerate/default_config.yaml`) yaml:

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

If you want to use `accelerate` to launch the training script with FSDP etc, instead of running `train.py` with `python train.py --args`, you must use `accelerate launch train.py --args`

## Examples

I have a [set of examples](examples/README.md) that show how to use the code for different tasks. The examples cover most functionality of the code. 

## Project Organization

### Inference

The entry point for inference is the [`infer.py`](infer.py) script. It supports both HuggingFace Transformers and vLLM inference pipelines for both Language Models and Vision Language Models. The call to inference has three components:
1. Core arguments: These are found in the click declaration of the function [`main`](infer.py) and should be passed in right after the filename with `python infer.py --modality vlm` etc
2. Framework selection: This is done with the `hf` or `vllm` command, which selects the HuggingFace Transformers or vLLM inference pipeline respectively. e.g. `python infer.py --model_name <name> hf`
3. Framework specific arguments: These are passed in after the `hf` or `vllm` command. For example, `python infer.py --model_name <name> hf --batch_size 8` will run inference with the HuggingFace Transformers pipeline. See the [huggingface](infer/huggingface_inference.py) and [vllm](infer/vllm_inference.py) inference files for the arguments that can be passed in after the `hf` or `vllm` command.

The scripts will expect your input to be a csv file with a column named `input` that contains the text to be processed (and `image` with a url or path to an image for VLMs). It also expects that the input file does *not* contain the columns `output` or `inference_completed`. The output will be saved in the same directory as the input file, with a suffix `_output` added to the filename, and as a json lines file (`.jsonl`). The names of the columns, as well as the output path can be changed with the appropriate arguments.  

This output file also automatically acts as a checkpoint if inference stops halfway, and unless you tell it not to, the code will always try to restart from a checkpoint. 

### Training

The entry point for training is the [`train.py`](train.py) script. The final model is always saved to `output_dir/final_checkpoint`

WandDB is used to log the metrics, and you can always recover the history of a prior run with:

```python
from utils import get_history
history = get_history("run_name")
```

#### Arguments
There are two sets of arguments this script accepts:
1. ScriptArguments: Check these out in the [`ScriptArguments`](train.py) class. 
2. Learning specific arguments: These depend on the kind of training you are doing, and all of them are taken from HuggingFace or HuggingFace TRL. Classification takes in the same arguments as [TrainingArguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py), Supervised Finetuning takes in the same arguments as [SFTConfig](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py) and Direct Preference Optimization takes in the same arguments as [DPOConfig](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py).

To see the parameters that can be used on the command line see the respective config files. All arguments that are used internally in a Trainer Class are passed on to the Trainer class. So, for example, if you want to set the number of epochs for training (for classification finetuning), you must add `--num_train_epochs <number>` to the set of args passed in. Essentially, go through the config file argument options and know that you can ignore any argument which has the following disclaimer: `This argument is not directly used by Trainer, it’s intended to be used by your training/evaluation scripts instead` (e.g. `--do_eval`). The only exception to this rule is `--resume_from_checkpoint` which takes in `<True/False/path-to-checkpoint>` and is used in the script. 

#### Checkpointing

By default, we do try to resume from a checkpoint. If the output directory is not found, we will begin training from the first step. If there is an output directory with no valid checkpoint, the code will *fail* unless --resume_from_checkpoint is `False`. When using LoRA + FSDP etc, the checkpoint files are *not* complete models, but rather sharded adapters, and cannot be read and treated as normal HuggingFace models. In order to use a saved checkpoint, you must relaunch the script, but set the `--num_train_epochs` or `--max_steps` value to be lower than the checkpoint. This way, the script will load the model and immediately save it. There's an example of this being done [here](examples/README.md#checkpointing).

#### Input Format
The essential format to follow for each training paradigm is given below:

1. Classification: input files must be `.csv` with `input` and `output` columns
2. Pretraining: input files can either be `.txt` or `.csv`, csv must have column `input` with the text to learn. You can also include an `output` column, in which case we will concatenate the two columns and pretrain on the whole thing. If instead, you want to only pretrain on the input column, pass in `--pretrain_with_output False`
3. Supervised Finetuning: input files must be a csv with `input` and `output` columns. Loss is only computed on completions/output, if you want loss to be computed on the input prompt as well, this is handled by the pretraining paradigm. 
4. Direct Preference Optimization: input files must be a csv with `input`, `chosen` and `rejected` columns.
5. Gradient Ascent: input files must be a csv with `input`, `output` and `forget` columns, where `forget` is a binary indicator as to whether or not that particular example should be forgotten. Setting `forget=0` for all rows is equivalent to running SFT
6. Negative Preference Optimization: input files must be a csv with `input` and `output` columns. 
