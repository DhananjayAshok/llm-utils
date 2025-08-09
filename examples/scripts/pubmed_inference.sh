source examples/scripts/env.sh
batch_size=20
file_names="qa_gen_train qa_gen_val qa_gen_background_train qa_gen_background_val"
for file_name in $file_names; do
  python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
  --max_new_tokens 200 hf --batch_size $batch_size --padding_side left
done