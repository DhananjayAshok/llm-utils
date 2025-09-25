source configs/config.env
accelerate launch train.py --training_kind clf --model_name meta-llama/Llama-3.2-1B-Instruct \
--output_dir $storage_dir/models/clf_model --num_train_epochs 10 \
--train_file $storage_dir/data/pubmedqa/hf_clf_train.csv --train_validation_split 0.85 --test_file $storage_dir/data/pubmedqa/hf_clf_val.csv --output_column label  \
--logging_strategy epoch --eval_strategy epoch \
--run_name pubmed-classification

echo "Training complete. Running inference on the test set."

python infer.py --model_name $storage_dir/models/clf_model/final_checkpoint --input_file $storage_dir/data/pubmedqa/hf_clf_val.csv hf --model_kind clf --batch_size 20

python3 << EOF
import pandas as pd; df = pd.read_json("$storage_dir/data/pubmedqa/hf_clf_val_output.jsonl", lines=True); print("Achieves Accuracy: ", (df.label == df.output).mean())
EOF