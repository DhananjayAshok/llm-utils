source configs/config.env
batch_size=20
file_name="test_qa"

model_name="meta-llama/Llama-3.2-1B-Instruct"
python infer.py --model_name $model_name --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
--output_file $storage_dir/data/pubmedqa/${file_name}_base_output.jsonl --max_new_tokens 200 hf --batch_size $batch_size --padding_side left

#python infer.py --model_name $model_name --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
#--output_file $storage_dir/data/pubmedqa/${file_name}_base_output.jsonl --max_new_tokens 200 vllm --max_model_len 4000

model_name="$storage_dir/models/ft_model/final_checkpoint"
python infer.py --model_name $model_name --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
--output_file $storage_dir/data/pubmedqa/${file_name}_ft_output.jsonl --max_new_tokens 200 hf --batch_size $batch_size --padding_side left
#python infer.py --model_name $model_name --input_file $storage_dir/data/pubmedqa/${file_name}.csv \
#--output_file $storage_dir/data/pubmedqa/${file_name}_ft_output.jsonl --max_new_tokens 200 vllm --max_model_len 4000


python3 << EOF
import pandas as pd
df_base = pd.read_json("$storage_dir/data/pubmedqa/test_qa_base_output.jsonl", lines=True)
df_ft = pd.read_json("$storage_dir/data/pubmedqa/test_qa_ft_output.jsonl", lines=True)
df_base["output"] = df_base["output"].apply(lambda x: x[0] if isinstance(x, list) else x)
df_ft["output"] = df_ft["output"].apply(lambda x: x[0] if isinstance(x, list) else x)
df_base["binary_output"] = df_base["output"].apply(lambda x: x.split("Conclusion:")[-1].strip().lower() if isinstance(x, str) else x)
df_ft["binary_output"] = df_ft["output"].apply(lambda x: x.split("Conclusion:")[-1].strip().lower() if isinstance(x, str) else x)
print("Base Rate: \n", df_base["answer"].value_counts(normalize=True)* 100)
df_base["correct"] = df_base["binary_output"] == df_base["answer"]
df_ft["correct"] = df_ft["binary_output"] == df_ft["answer"]
print("Base LLama3-Instruct Model Achieves PubmedQA Accuracy: ", (df_base.groupby("answer")['correct'].mean()*100))
print("Fine-tuned LLama3-Instruct Model Achieves PubmedQA Accuracy: ", (df_ft.groupby("answer")['correct'].mean()*100))
EOF