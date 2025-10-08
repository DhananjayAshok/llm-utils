folders=("clf" "sft" "pre" "dpo" "ga" "npo")
for folder in "${folders[@]}"; do
    file_name="tests/training/$folder/lm.sh"
    bash "$file_name"
done