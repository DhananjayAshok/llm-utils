bash tests/training/pre/lm.sh
folders=("clf" "sft" "dpo" "ga" "npo")
for category in "${folders[@]}"; do
    bash tests/training/$category/lm.sh
    #bash tests/training/$category/vlm.sh
done
bash tests/training/checkpointing.sh
