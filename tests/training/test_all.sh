bash tests/training/pre/lm.sh
for category in clf sft dpo; do
    bash tests/training/$category/lm.sh
    bash tests/training/$category/vlm.sh
done
