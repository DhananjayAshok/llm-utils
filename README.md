# Language Model Utilities

Useful code for training and inference of Language Models. I currently support the following functionality:

Language Models:
1. Inference with HuggingFace Transformers and vLLM (no vLLM support for VLMs at the moment)
2. Pretraining
3. Finetuning (Classification and Supervised Finetuning for Generation)
4. Preference Optimization (Direct Preference Optimization, Contrastive Preference Optimization)
5. Unlearning (Gradient Ascent, Negative Preference Optimization)

All code is based on HuggingFace Transformers and TRL and supports FSDP with multiple GPUs. 

This branch is the stable version of the code base and does not support running the examples tests / actively adding features. If you want to do that, see the dev branch instead. 

## Setup

This branch is meant to be used as a submodule in a higher-level project that will call on its functionalities. We assume that you *already* have a GitHub Repository set up and want to set up llm-utils inside it. 

First, add the repo as a submodule and update the repo:
```bash
git submodule add -b app <url_to_this_repo>
git submodule init
git submodule update
```

Then follow the [instructions](setup/README.md) to set up the environment with the right packages and Python version. That's all the setup you need for inference, but for training, you will need to log in to WandB with 

```bash
wandb login
```

### FSDP (Optional)
The code base supports FSDP, but it can be a bit buggy. See the dev repo for instructions on setting it up. 


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
