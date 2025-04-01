# Language Model Utilities

Useful code for training and inference of Language Models. 

## Examples

To try out the various training and inference codes, you can load in sample data with (make sure to first set the relevant parameters in the config directories yaml files):

```bash
python examples/generate_example_data.py
```

You can specify the specific variants you want with

```bash
python examples/generate_example_data.py get_example_data --variants sft --variants pref
```

Available variants are: 
1. pre: Pretraining
2. sft: Supervised Finetuning (text)
3. clf: Classification 
4. pref: Preference Optimization

Then, run the script you want to train with using:

```bash
python examples/scripts/sft.sh
```

## Project organization

The inference code: 

The training code is essentially a quick wrapper around HuggingFace Trainer and TRL. In general the only required arguments are training_kind, model_name and a train_file for data. The call looks like

```bash
python train.py --training_kind <pre/sft/clf/dpo> --model_name <name> --output_dir tmp/
```




The essential format to follow for each training paradigm is given below:

1. Pretraining: input files can either be .txt or .csv, csv must have column input with the text to learn. If you want to do supervised finetuning but fit loss on the prompt as well, then this is handled by this training mode. In this case you must specify a csv file with input and output columns, and make sure the relevant ScriptArgument parameter is set to true
2. Supervised Finetuning: input files must be a csv with input and output columns. Loss is only computed on completions
3. Direct Preference Optimization: input files must be a csv with input, chosen and rejected columns. 