# Language Model Utilities

Useful code for training and inference of Language Models.

## Sample Data

To try out the various training and inference codes, you can load in sample data with:

```bash
python main.py get_example_data
```

You can specify the specific variants you want with

```bash
python main.py get_example_data --variant sft --variant pref
```

Available variants are: 
1. pre: Pretraining
2. sft: Supervised Finetuning (text)
3. clf: Classification 
4. pref: Preference Optimization