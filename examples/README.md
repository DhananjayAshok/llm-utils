# Examples

Once you've followed the environment [setup instructions](../README.md#setup) (especially the accelerate config setup steps for training), add the `storage_dir` variable to the example [environment file](scripts/env.sh). 

You can now start running these example experiments to get a hang of this repo and its functionalities. 

Any and all commands should be run from the root of the repo. 

## Knowledge Acquisition with Language Models

### Overview:
This guided example will take you through using the following features of this repo:
1. LM Inference
2. LM Pretraining
3. LM Finetuning
4. LM Preference Tuning

The example centers around the PubMedQA dataset, which is a question answering dataset based on scientific, biomedical articles. The goal is to train a language model to acquire the knowledge in the articles, so that it can later answer the questions in without having to look at the articles again. To do this, we'll implement the following pipeline:

1. Synthetically generate question answer pairs from the articles. We hope that these will be similar-ish to the test questions, and that learning these will allow the LM to answer the test queries better.
2. Paraphrase the generated question answer pairs. 
3. Fine-tune a question answering model on the paraphrases, while keeping the original LM generations as a validation set. By maintaining a validation set, we can stop training if we start overfitting to the specific wording of the questions in the train set. 

We'll be using the Llama3-8B model for generation and LLama3-1B for fine-tuning, but you can use any other model that is compatible with the HuggingFace Transformers library.


### Setup
Start by setting up the data with:
```bash
python create_examples.py setup --dataset_names pubmedqa
```

This will create a few files in the `$storage_dir/data/pubmedqa` directory:
- `test_qa.csv`: The question, answer pairs that we will be testing our eventual models on
- `pretraining.csv`: The text of all the articles in PubMedQA, which we will use for pretraining to make the LM more familiar with the biomedical domain. 
- `qa_gen_standard.csv`: Contains articles, and prompts that get a LM to generate question answer pairs from the articles. We will use this to generate synthetic QA pairs for finetuning a LM.
- `qa_gen_method.csv`: Quite similar to the above, but the prompts incentivize the model to only generate questions on the study, as opposed to the knowledge inferable from it. We will use this and the data generated from the previous file to preference tune a LM that only asks study related questions. 

### Inference

The first step is to run inference on the qa_gen files to generate the synthetic QA pairs. This is done with the following command (don't run it just yet):
```bash
python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/qa_gen_standard.csv --max_new_tokens 150 hf 
```
You may get the warning message: 
```bash
A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
```

In that case, add the `--padding_side left` argument to the command above, after the `hf` command.

This will trigger the huggingface inference pipeline, which has the following arguments you can pass in after the `hf` command
- `--model_kind`: whether the model is a classification model (clf) or a generative model (gen, default.)
- `--batch_size`: duh
- `--quantization`: whether to use a form of quantization for the weights
- `--cache_implementation`: see [Cache Options](https://huggingface.co/docs/transformers/en/kv_cache)
- `--cache_prefix`: should we [prefill the cache with the prefix](https://huggingface.co/docs/transformers/en/kv_cache#prefill-a-cache). Should speed up generation for large models.

There are other options for tracking perplexity, see [the click options](../inference/huggingface_inference.py#204) for more. 

#### Inferring Batch Size
But before running inference, let's  identify the largest batch size we can use. To do that, we will fix some generation configurations, so our code knows how we plan on running the model:
- `max_new_tokens`: 200
- `num_return_sequences`: 5
- `num_beams`: 5
- `num_beam_groups`: 5

This should get a nice variety of questions from each article. 

Now let's run the following command to find the largest batch size we can use:
```bash
python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/qa_gen_standard.csv --max_new_tokens 200 --num_return_sequences 5 infer_batch_size --num_beams 5
```

This gives me the recommended maximum batch size of 6. If you get 0 (i.e. nothing works), try setting `cache_implementation=offloaded` and try again.

We'll pick a batch_size slightly smaller than the maximum: 4. 

To run inference on all of the generation files, you can use the following command (make sure to set your desired batch_size first): 
```bash
bash examples/scripts/pubmed/pubmed_inference.sh
```

Note, you do *not* need to run the above line in order to proceed with the tutorial. I have precomputed the output and make them available from my HuggingFace repo. To set them up on your system, run the command:
```bash
python create_examples.py pubmed_process
```
This will set up the fine-tuning csv's with columns `input` (the question text) and `output` (the long answer + conclusion text). it also set's up a classification csv pair with columns `input` (the question text) and `label` (1 if query is results based, 0 if it is background based)

### Classification Finetuning
We'll use the files from above and train a classifier that tells us whether a generated question is a results based (1) or background based (0) question. First, make sure you have `CUDA_HOME` accessible in the environment where the script will execute. Then, run:

```bash
bash examples/scripts/pubmed/pubmed_clf.sh
```

This command triggers classification fine-tuning of a small LLama3 model with:
```bash
accelerate launch train.py --training_kind clf --model_name meta-llama/Llama-3.2-1B-Instruct \
--output_dir $storage_dir/models/clf_model --num_train_epochs 10 \
--train_file $storage_dir/data/pubmedqa/hf_clf_train.csv --train_validation_split 0.85 --test_file $storage_dir/data/pubmedqa/hf_clf_val.csv --output_column label  \
--logging_strategy epoch --eval_strategy epoch \
--run_name pubmed-classification
```

Then, it uses that saved model (saves to output_dir/final_checkpoint) to conduct inference on the test file:
```bash
python infer.py --model_name $storage_dir/models/clf_model/final_checkpoint --input_file $storage_dir/data/pubmedqa/hf_clf_val.csv hf --model_kind clf --batch_size 20
```

As you can see, we achieve significantly higher than random accuracy on the test set. 


#### Inferring Batch Size

Unlike with inference, we do not have any script to infer the batch size. Instead, just start running and go to the config file (it should be printed in the log) to see what your current batch size is (by default, `per_device_train_batch_size=8`). While the code is running, go to the WanDB panel and check out the System Report: GPU Memory Allocated (%). You should ideally be around 90% allocation, and if you are lower, scale your batch size appropriately and re-run the script. 

#### Checkpointing
You can interupt training at any time with `Ctrl+C` or by sending a kill signal to the process, and if you have set reasonable values for `--save_strategy` and `--save_every`, you will have saved checkpoints. 

Let's say you run the above script, and decide to stop it manually at epoch 5. then to restart and go all the way to 10 epochs, you just need to re-run the script with the `--restore_from_checkpoint True` argument. This has a chance of failing if you interupted it in the middle of saving the checkpoint, leading to the latest checkpoint being corrupted and hence not readable. You can check if this has happened by looking into the checkpoint directory. If it does not have the file `trainer_state.json`, then the checkpoint is corrupted. Delete this folder, and the code will fall back to the previous valid checkpoint. 



If you are using LoRA with FSDP etc, these checkpoints are not ready-to-go HuggingFace models, so while you can restart training with them, you cannot use them outside the training script. To make them usable, you should re-run the training script, except set the `--num_train_epochs` or `--max_steps` value to be something lower than the checkpoint's value. That way, it will just load and save a final model. So, continuing off the example above, you can use the checkpoint at epoch 5 with:
```bash
accelerate launch train.py --training_kind clf --model_name meta-llama/Llama-3.2-1B-Instruct \
--output_dir $storage_dir/models/clf_model --num_train_epochs 5 --restore_from_checkpoint True \
--train_file $storage_dir/data/pubmedqa/hf_clf_train.csv  --test_file $storage_dir/data/pubmedqa/hf_clf_val.csv --output_column label  \
``` 



### Supervised Finetuning
To train a Question Answering model with supervised finetuning, run:
```bash
bash examples/scripts/pubmed/pubmed_sft.sh
```

This calls on:
```bash
accelerate launch train.py --training_kind sft --model_name meta-llama/Llama-3.2-1B-Instruct \
--output_dir $storage_dir/models/ft_model \
--train_file $storage_dir/data/pubmedqa/hf_ft_train.csv --validation_file $storage_dir/data/pubmedqa/hf_ft_val.csv \
--num_train_epochs 50 \
--per_device_train_batch_size 24 --per_device_eval_batch_size 24 \
--learning_rate 1e-4 --weight_decay 0.1 \
--logging_strategy steps --logging_steps 200 \
--eval_strategy steps --eval_steps 200 \
--save_strategy steps --save_steps 200 \
--early_stopping_patience 5 \
--load_best_model_at_end True \
--run_name pubmed-sft
```

### Pretraining

Almost the exact same as above, just change the `training_kind` to `pre` and change the files accordingly. Pretraining doesn't have a validation file, so just set a `train_validation_split`. 


### Preference Optimization



## Question Generation with Vision Language Models
This example covers:
1. VLM Inference
2. VLM Finetuning
3. VLM Preference Tuning
