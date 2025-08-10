source examples/scripts/env.sh
batch_size=20
file_names="qa_gen_standard qa_gen_method test_qa"
for file_name in $file_names; do
  python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
  --max_new_tokens 200 hf --batch_size $batch_size --padding_side left
  #python infer.py --model_name meta-llama/Llama-3.1-8B-Instruct --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
  #--max_new_tokens 200 vllm --max_model_len 4000
done

python3 << EOF
import pandas as pd; df = pd.read_json("$storage_dir/data/pubmedqa/test_qa_output.jsonl", lines=True)
df["output"] = df["output"].apply(lambda x: x[0] if isinstance(x, list) else x)
df["binary_output"] = df["output"].apply(lambda x: x.split("Conclusion:")[-1].strip().lower() if isinstance(x, str) else x)
print("Base Rate: \n", df["final_decision"].value_counts(normalize=True)* 100)
df["correct"] = df["binary_output"] == df["final_decision"]
print("LLama3-Instruct Model Achieves PubmedQA Accuracy: ", (df.groupby("final_decision")['correct'].mean()*100))
EOF