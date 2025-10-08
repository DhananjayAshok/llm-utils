#!/bin/bash
source configs/config.env
source tests/training/common.env
export common_line="--lora_target_modules q_proj --num_train_epochs 5  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --train_validation_split 0.85 --validation_test_split 0.5 --logging_strategy epoch --eval_strategy epoch --save_strategy epoch --save_steps 1 --logging_steps 0.5 --eval_steps 0.5 --load_best_model_at_end"
# --- 1. Define Variables ---
test_lm=${test_lms[0]}
PYTHON_SCRIPT="train_launch train.py --training_kind clf --model_name $test_lm \
    --output_dir $storage_dir/models/tmp_clf_model \
    --train_file tmp_test_data/tmp_clf.csv --output_column label  \
    --run_name test-clf-$test_lm $common_line"
LOG_FILE="tmp.log"
SEARCH_STRING="'epoch': 2.0"

# --- 2. Run Python Script in the Background and Redirect Output ---

# Ensure log file is clean before starting
> $LOG_FILE

# Run the Python script in the background and pipe its stdout/stderr to a log file
# The '$$' gives us the PID of the current BASH script, which we use as a prefix
# for the log file to make it unique if running multiple times.
echo "Testing checkpoint functionality. Running Python script in the background. Output will be logged to $LOG_FILE"
$PYTHON_SCRIPT &> $LOG_FILE &

# Store the Process ID (PID) of the backgrounded Python command
PYTHON_PID=$!

echo "Python PID: $PYTHON_PID"
echo "Watching log file: $LOG_FILE for string: '$SEARCH_STRING'. Will kill once found..."

# --- 3. Monitor the Log File ---

# Use 'tail -f' to continuously stream the log file
# Use 'grep -q' to silently search for the string
# 'while ! ... ; do ... done' loops until the command (grep) succeeds (finds the string)
while ! grep -q "$SEARCH_STRING" $LOG_FILE; do
    # Sleep briefly to avoid constantly checking the file, saving CPU cycles
    sleep 0.5
    # Check if the Python process is still running (it might have finished naturally)
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo "✅ Python script finished naturally before string was found. Check $LOG_FILE for details."
        exit 0
    fi
done

# --- 4. Kill the Process When String is Found ---

echo "---"
echo "🛑 Found string: '$SEARCH_STRING'!"

# Use 'kill' to stop the process. SIGTERM (default) is a graceful termination signal.
kill $PYTHON_PID

# Wait a moment for the process to die
sleep 1

# Check if the process is actually dead. If not, forcefully kill it with SIGKILL (-9).
if kill -0 $PYTHON_PID 2>/dev/null; then
    echo "⚠️ Process did not terminate gracefully. Sending SIGKILL (-9)..."
    kill -9 $PYTHON_PID
fi

echo "✅ Python process $PYTHON_PID has been successfully terminated."

# Optionally, print the last few lines of the log for context
echo "--- Last few lines of log ---"
tail -n 5 $LOG_FILE

echo "Running training command once more. Should read from last checkpoint and complete training."
$PYTHON_SCRIPT
rm $LOG_FILE
rm -rf $storage_dir/models/tmp_clf_model
echo "Check the output above, it should have detected and resumed from a checkpoint."