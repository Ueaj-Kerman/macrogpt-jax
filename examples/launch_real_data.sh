#!/bin/bash

# Test distributed real data loading with 2 processes, 1 device each

export JAX_PLATFORMS=cpu
export JAX_NUM_CPU_DEVICES=1

NUM_PROCS=2
HOST=${HOST:-"localhost:10003"}

echo "Launching distributed real data examples with $NUM_PROCS processes..."
echo ""

# Launch processes in background
for i in $(seq 0 $((NUM_PROCS - 1))); do
    echo "Starting process $i"
    (
        WORLD_SIZE=$NUM_PROCS \
        RANK=$i \
        HOST=$HOST \
        PYTHONPATH=/mnt/c/Users/devse/IdeaProjects/nanollama \
        ./scripts/run_python.sh -u examples/distributed_real_data.py 2>&1 | sed "s/^/[P$i] /"
    ) &
done

# Wait for all background processes to complete
wait

echo ""
echo "Examples complete!"
