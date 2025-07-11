#!/bin/bash

# Find and kill all train.py processes
pids=$(ps aux | grep -E "[p]ython.*train\.py" | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No train.py processes found."
else
    echo "Found train.py process(es) with PID(s): $pids"
    for pid in $pids; do
        echo "Killing process $pid..."
        kill $pid
    done
    echo "All train.py processes killed."
fi
