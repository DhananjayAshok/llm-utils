# Testing

You can just run all tests with:
```bash
bash tests/test_all.sh
```
Alternatively, you can run each segment individually. First, run the command to set up the datasets we'll be using for all subsequent tests (all commands to be run from root):
```bash
bash tests/example_data/test_all.sh
```
This should set up all the required datasets. If it fails, check that you have set up the environment properly and you have set up the configs folder properly. 

Then, test the inference functionality:

```bash
bash tests/inference/test_all.sh
```
This code typically should not be failing. If you get OOM issues then consider reducing the batch size or changing the models in `tests/inference/common.env` smaller versions. Of course, if you get an error saying that you are trying to access a gated model, then you will have to agree to the LLama terms of use and add your huggingface credentials with:
```bash
huggingface-cli login
```
Finally, test the training functionality:
```bash
bash tests/training/test_all.sh
```
This will fail if you have not run `accelerate config`. If you have, and your model doesn't fit onto the GPU with FSDP etc., then consider removing accelerate multiprocessing by editing the [function](tests/training/common.env) as explained in the env file. 
