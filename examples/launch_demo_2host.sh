#!/bin/bash
# Launch script for 2-host distributed demo (2 data × 2 tensor mesh)

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Launching 2-host distributed demo..."
echo "Mesh: 2 data × 2 tensor (4 devices total)"
echo ""

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT"
export JAX_PLATFORMS=cpu
export DIST_AUTO=False
export WORLD_SIZE=2
export HOST="localhost:1234"

# Launch both processes in background
echo "Starting Host 0 (rank 0)..."
RANK=0 "$PROJECT_ROOT/scripts/run_python.sh" "$SCRIPT_DIR/demo_distributed_complete.py" --orientation 2 > /tmp/demo_host0.log 2>&1 &
PID0=$!

echo "Starting Host 1 (rank 1)..."
RANK=1 "$PROJECT_ROOT/scripts/run_python.sh" "$SCRIPT_DIR/demo_distributed_complete.py" --orientation 2 > /tmp/demo_host1.log 2>&1 &
PID1=$!

# Wait for both to complete
echo "Waiting for processes to complete..."
wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?

echo ""
echo "==================================================================="
echo "HOST 0 OUTPUT:"
echo "==================================================================="
cat /tmp/demo_host0.log

echo ""
echo "==================================================================="
echo "HOST 1 OUTPUT:"
echo "==================================================================="
cat /tmp/demo_host1.log

echo ""
if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo "✓ Both hosts completed successfully!"
    exit 0
else
    echo "✗ Some hosts failed (exit codes: $EXIT0, $EXIT1)"
    exit 1
fi
