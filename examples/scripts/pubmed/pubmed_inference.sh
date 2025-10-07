source configs/config.env
batch_size=20
file_names="qa_gen_standard qa_gen_method"
for file_name in $file_names; do
  #python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
  #--max_new_tokens 200 --ignore_checkpoint hf --batch_size $batch_size --padding_side left
  python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
  --max_new_tokens 200 vllm --max_model_len 4000
done

exit 

python3 << EOF
import pandas as pd
df = pd.read_csv("/project/jonmay_1426/ashokd//tmp/data/pubmedqa/qa_gen_standard.csv")
df.sample(1000, random_state=42).to_csv("tmp_inf.csv", index=False)
EOF

python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file tmp_inf.csv \
--max_new_tokens 200 --ignore_checkpoint vllm --max_model_len 4000