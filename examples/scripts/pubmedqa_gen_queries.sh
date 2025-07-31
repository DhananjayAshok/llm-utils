storage_dir= # whatever you set in configs/private_vars.yaml
if [ -z "$storage_dir" ]; then
  echo "Please set the storage_dir variable in configs/private_vars.yaml"
  exit 1
file_names="qa_gen_train qa_gen_val qa_gen_background_train qa_gen_background_val"
for file_name in $file_names; do
  python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
  --max_new_tokens 350 hf --cache_implementation offloaded --batch_size 10 --cache_prefix False
done