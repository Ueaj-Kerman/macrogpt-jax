#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
optimizers=("muon" "multiscale")
lrs=(0.025 0.03 0.035 0.04)
instances_per_config=1

# Arrays to store job info
pids=()
run_names=()
gpu=0

echo "Starting optimized learning rate sweeps"
echo "========================================"
echo "Optimizers: ${optimizers[@]}"
echo "Learning rates: ${lrs[@]}"
echo "Instances per config: $instances_per_config"
echo "Total jobs: 8 (using 8 GPUs)"
echo "Logs directory: logs/"
echo "Models directory: models/"
echo "========================================"

# Launch all 8 jobs concurrently
for optimizer in ${optimizers[@]}; do
    for lr in ${lrs[@]}; do
        for instance in $(seq 1 $instances_per_config); do
            run_name="${optimizer}_lr${lr}_run${instance}_$(date +%Y%m%d_%H%M%S)"
            run_names+=("$run_name")
            
            echo "Starting $optimizer with lr=$lr on GPU $gpu (run: $run_name)"
            
            # Create models directory if it doesn't exist
            mkdir -p models
            
            # Run training in background, assigning one GPU per job
            CUDA_VISIBLE_DEVICES=$gpu BASE_LR=$lr OPTIMIZER=$optimizer RUN_NAME=$run_name \
                MODEL_PATH="models/${run_name}.safetensors" \
                python -m ueaj.train.train > logs/${run_name}.log 2>&1 &
            
            # Store PID
            pids+=($!)
            gpu=$((gpu + 1))
        done
    done
done

echo ""
echo "All 8 jobs launched. Waiting for completion..."
echo "----------------------------------------------"

# Wait for all jobs to complete
for i in ${!pids[@]}; do
    pid=${pids[$i]}
    run_name=${run_names[$i]}
    wait $pid
    status=$?
    if [ $status -eq 0 ]; then
        echo "✓ Job $run_name (PID $pid) completed successfully"
    else
        echo "✗ Job $run_name (PID $pid) failed with status $status"
        # Show last few lines of error log
        log_file="logs/${run_name}.log"
        if [ -f "$log_file" ]; then
            echo "  Error output (last 10 lines):"
            tail -n 10 "$log_file" | sed 's/^/    /'
            echo "  ----------------------------------------"
        fi
    fi
done

echo ""
echo "All sweeps completed!"
echo "===================="

# Summary
echo ""
echo "Summary:"
echo "--------"
successful=0
failed=0
for i in ${!pids[@]}; do
    run_name=${run_names[$i]}
    log_file="logs/${run_name}.log"
    if [ -f "$log_file" ]; then
        if grep -q "Traceback\|Error\|error\|Failed" "$log_file" 2>/dev/null; then
            echo "✗ Failed: $run_name"
            failed=$((failed + 1))
        else
            echo "✓ Success: $run_name"
            successful=$((successful + 1))
        fi
    fi
done

echo ""
echo "Total: $successful successful, $failed failed out of 8 jobs"

# Check for saved models
echo ""
echo "Saved models:"
echo "-------------"
if [ -d "models" ]; then
    for model_file in models/*.safetensors; do
        if [ -f "$model_file" ]; then
            size=$(du -h "$model_file" | cut -f1)
            echo "✓ $(basename $model_file) ($size)"
        fi
    done
else
    echo "No models directory found"
fi