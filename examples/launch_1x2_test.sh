#!/bin/bash
# Launch script for 1×2 mesh test (1 data × 2 tensor)

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Testing 1 Data × 2 Tensor Mesh"
echo "Expected: Only 1 host loads data, other creates zeros"
echo ""

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT"
export JAX_PLATFORMS=cpu
export DIST_AUTO=False
export WORLD_SIZE=2
export HOST="localhost:1234"

# Launch both processes in background
echo "Starting Host 0 (rank 0) - should LOAD data..."
RANK=0 "$PROJECT_ROOT/scripts/run_python.sh" "$SCRIPT_DIR/test_1x2_mesh.py" > /tmp/test_1x2_host0.log 2>&1 &
PID0=$!

echo "Starting Host 1 (rank 1) - should create ZEROS..."
RANK=1 "$PROJECT_ROOT/scripts/run_python.sh" "$SCRIPT_DIR/test_1x2_mesh.py" > /tmp/test_1x2_host1.log 2>&1 &
PID1=$!

# Wait for both to complete
echo "Waiting for processes to complete..."
wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?

echo ""
echo "==================================================================="
echo "HOST 0 OUTPUT (Should LOAD data):"
echo "==================================================================="
cat /tmp/test_1x2_host0.log

echo ""
echo "==================================================================="
echo "HOST 1 OUTPUT (Should create ZEROS):"
echo "==================================================================="
cat /tmp/test_1x2_host1.log

echo ""
if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo "✓ Test passed! Check that Host 0 loaded and Host 1 created zeros."
    exit 0
else
    echo "✗ Test failed (exit codes: $EXIT0, $EXIT1)"
    exit 1
fi
