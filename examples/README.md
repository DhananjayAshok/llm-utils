# Examples

Once you've followed the environment [setup instructions](README.md), you can start running these example experiments to get a hang of this repo and its functionalities. 

Any and all commands should be run from the root of the repo. 

## Knowledge Acquisition with Language Models

### Overview:
This guided example will take you through using the following features of this repo:
1. LM Inference
2. LM Pretraining
3. LM Finetuning
4. LM Preference Tuning

The example centers around the PubMedQA dataset, which is a question answering dataset based on scientific, biomedical articles. The goal is to train a language model to acquire the knowledge in the articles, so that it can later answer the questions in without having to look at the articles again.

We'll be using the Qwen3-8B model for this example, but you can use any other model that is compatible with the HuggingFace Transformers library.


### Setup
Start by setting up the data with:
```bash
python create_examples.py setup --dataset_names pubmedqa
```

This will create a few files in the `$storage_dir/data/pubmedqa` directory:
- `test_qa.csv`: The question, answer pairs that we will be testing our eventual models on
- `pretraining.csv`: The text of all the articles in PubMedQA, which we will use for pretraining to make the LM more familiar with the biomedical domain. 
- `qa_gen_[train/val].csv`: Contains articles, and prompts that get a LM to generate question answer pairs from the articles. We will use this to generate synthetic QA pairs for finetuning a LM.
- `qa_gen_background_[train/val].csv`: Quite similar to the above, but the prompts incentivize the model to only generate questions on the background or premise of the article, as opposed to its results. We will use this and the data generated from the previous file to preference tune a LM that only asks background related questions. 

The first step is to run inference on the qa_gen files to generate the synthetic QA pairs. This is done with the following command:
```bash
python infer.py --model_name Qwen/Qwen3-8B --input_file $storage_dir/data/pubmedqa/qa_gen_train.csv --max_new_tokens 256 hf
```
This will trigger the huggingface inference pipeline, which has the following arguments you can pass in after the `hf` command
- `--model_kind`: whether the model is a classification model (clf) or a generative model (gen, default.)
- `--batch_size`: duh
- `--quantization`: whether to use a form of quantization for the weights
- `--cache_implementation`: see [Cache Options](https://huggingface.co/docs/transformers/en/kv_cache)
- `--cache_prefix`: should we [prefill the cache with the prefix](https://huggingface.co/docs/transformers/en/kv_cache#prefill-a-cache). Should speed up generation for large models.

There are other options for tracking perplexity, see [the click options](inference/huggingface_inference.py) for more. 


## Question Generation with Vision Language Models
This example covers:
1. VLM Inference
2. VLM Finetuning
3. VLM Preference Tuning