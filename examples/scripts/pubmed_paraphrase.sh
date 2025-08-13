source examples/scripts/env.sh
batch_size=30
standard_args="--max_new_tokens 200 --model_name meta-llama/Llama-3.1-8B-Instruct --ignore_checkpoint --num_return_sequences 5 --do_sample hf --batch_size $batch_size --padding_side left"

python infer.py --input_file $storage_dir/data/pubmedqa/standard_paraphrase.csv \
--input_column question_input --output_file $storage_dir/data/pubmedqa/standard_paraphrase_question_output.jsonl \
$standard_args