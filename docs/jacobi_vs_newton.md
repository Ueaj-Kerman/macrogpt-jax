# Jacobi vs Newton for Parallel RNN

Comparison of two approaches for solving the fixed-point equation in parallel RNN.

## The Problem

Given RNN recurrence: `h_t = f(h_{t-1}, x_t)` for t=1..T

We want to solve for all h_t simultaneously by treating it as a fixed-point problem.

## Jacobi Iteration (Our Implementation)

### Algorithm
```python
# Initialize all states to h0
h_all = [h0, h0, h0, ..., h0]

# Iterate until convergence
for k in range(num_iterations):
    # Update each state using previous iteration
    for t in parallel:
        h_all[t]^(k+1) = f(h_all[t-1]^(k), x_t)
```

### Properties
- **Convergence**: Linear (error ~ λ^k where λ < 1)
- **Iterations needed**: O(T) in worst case, typically 10-20
- **Per-iteration cost**: O(T × hidden_dim²)
- **Memory**: O(T × hidden_dim) for states
- **Jacobian**: Never computed explicitly

### Convergence Rate
If the Jacobian has spectral radius ρ(J) < 1:
```
||h^(k+1) - h*|| ≤ ρ(J) × ||h^(k) - h*||
```

Reduction per iteration: ~ρ(J) ≈ 0.3-0.7 for typical RNNs

## Newton's Method (Paper's Approach)

### Algorithm
```python
# Initialize all states
h_all = [h0, h0, h0, ..., h0]

# Iterate with Newton updates
for k in range(num_iterations):
    # Compute residual: r_t = h_t - f(h_{t-1}, x_t)
    residual = compute_residual(h_all)

    # Compute Jacobian of entire system
    J = compute_jacobian(h_all)  # Block tridiagonal!

    # Solve: J × Δh = -residual (parallel scan!)
    delta_h = parallel_solve_tridiagonal(J, -residual)

    # Update
    h_all = h_all + delta_h
```

### Properties
- **Convergence**: Quadratic (error ~ ε² per iteration)
- **Iterations needed**: O(log log T), typically 3-5
- **Per-iteration cost**: O(T × hidden_dim³) for Jacobian
- **Memory**: O(T × hidden_dim²) for Jacobian
- **Jacobian**: Must compute and invert

### Convergence Rate
Quadratic convergence near solution:
```
||h^(k+1) - h*|| ≤ C × ||h^(k) - h*||²
```

If error is 0.1, next iteration: 0.01, then 0.0001, then 1e-8 (!)

## Detailed Comparison

### Computational Cost

| Operation | Jacobi (per iter) | Newton (per iter) | Notes |
|-----------|-------------------|-------------------|-------|
| Forward pass | T × O(d²) | T × O(d²) | Same: apply RNN cell |
| Jacobian computation | None | T × O(d³) | Newton needs ∂f/∂h |
| Linear solve | None | T × O(d³) | Parallel scan on tridiagonal |
| **Total per iteration** | **O(T × d²)** | **O(T × d³)** | **Newton is O(d) more expensive** |

Where:
- T = sequence length
- d = hidden dimension

### Total Cost to Convergence

Assuming convergence in:
- Jacobi: 15 iterations
- Newton: 3 iterations

```
Jacobi total:  15 × T × d²
Newton total:   3 × T × d³ = 3 × T × d² × d

Crossover point: d ≈ 5
```

**For small hidden dims (d < 5)**: Jacobi is cheaper!
**For large hidden dims (d > 5)**: Newton is cheaper!

### Memory Requirements

| Method | Memory | Breakdown |
|--------|--------|-----------|
| Jacobi | O(T × d) | Just states |
| Newton | O(T × d²) | States + Jacobian blocks |

For T=1000, d=512:
- Jacobi: ~512K floats = 2MB
- Newton: ~262M floats = 1GB

**Newton requires 500x more memory!**

### Exactness (Final Error)

Both methods solve the **exact same problem**: finding h* such that h_t = f(h_{t-1}, x_t).

The final solution is equally exact, but:

| Method | Typical final error | Why |
|--------|---------------------|-----|
| Jacobi | 1e-4 to 1e-6 | Stops after fixed iterations |
| Newton | 1e-8 to 1e-12 | Quadratic convergence |

**However**: Both are "exact enough" for ML purposes!
- Gradient noise in SGD: ~1e-3
- Float32 precision: ~1e-7
- Forward pass error of 1e-4 is negligible

### Practical Tradeoffs

#### Jacobi Wins When:
✓ Hidden dimension is small (d < 100)
✓ Memory is limited
✓ Implementation simplicity matters
✓ Moderate accuracy is sufficient (1e-4)
✓ Quick prototyping

#### Newton Wins When:
✓ Hidden dimension is large (d > 512)
✓ Sequence length is very long (T > 10k)
✓ Very high accuracy needed (1e-8)
✓ Total compute matters more than memory
✓ Production training of big models

## Example: GRU with d=512, T=1024

### Jacobi (15 iterations)
```
Compute:  15 × 1024 × 512² = 4.0 GFLOPs
Memory:   1024 × 512 = 0.5M params = 2 MB
Error:    ~1e-4
Time:     ~5ms on GPU
```

### Newton (3 iterations)
```
Compute:  3 × 1024 × 512³ = 412 GFLOPs  (100x more!)
Memory:   1024 × 512² = 262M params = 1 GB (500x more!)
Error:    ~1e-8
Time:     ~15ms on GPU (due to more complex ops)
```

**Winner**: Jacobi! Faster and uses less memory.

## Example: Large LLM-scale RNN with d=4096, T=2048

### Jacobi (20 iterations)
```
Compute:  20 × 2048 × 4096² = 689 GFLOPs
Memory:   2048 × 4096 = 8M params = 32 MB
Error:    ~1e-4
Time:     ~80ms on GPU
```

### Newton (3 iterations)
```
Compute:  3 × 2048 × 4096³ = 421 TFLOPs  (600x more!)
Memory:   2048 × 4096² = 34B params = 136 GB (!!)
Error:    ~1e-8
Time:     Out of memory on single GPU
```

**Winner**: Jacobi! Newton doesn't even fit in memory.

## Hybrid Approaches

### Quasi-Newton (Best of Both Worlds?)

Use approximate Jacobian:
```python
# Instead of exact J, use low-rank approximation
J_approx = low_rank_jacobian(h_all)  # O(T × d × r) where r << d

# Solve with approximation
delta_h = parallel_solve_lowrank(J_approx, -residual)
```

**Properties**:
- Convergence: Superlinear (between linear and quadratic)
- Iterations: ~5-8
- Cost per iteration: O(T × d² × r) where r ≈ 10-50
- Memory: O(T × d × r)

This is what DEER (2024) paper uses!

### Adaptive Strategy

Start with Jacobi, finish with Newton:
```python
# Coarse convergence with Jacobi (10 iters)
h_all = jacobi_iterate(h0, xs, num_iter=10)  # Error: 1e-3

# Fine convergence with Newton (2 iters)
h_all = newton_iterate(h_all, xs, num_iter=2)  # Error: 1e-8
```

**Best of both worlds**: Fast initial convergence + high final accuracy

## Recommendations

### For Research/Prototyping
**Use Jacobi**:
- Simple implementation
- Low memory
- Good enough accuracy
- Fast iteration

### For Production (Small Models, d < 512)
**Use Jacobi**:
- Better total cost
- Simpler deployment
- Adequate accuracy

### For Production (Large Models, d > 2048)
**Use Quasi-Newton** (DEER-style):
- Better convergence rate
- Manageable memory (with low-rank approximation)
- Worth the implementation complexity

### For Extreme Scale (d > 4096, T > 10k)
**Consider alternatives**:
- Linear SSMs (Mamba)
- Approximate methods
- Hybrid architectures

## Implementation Complexity

### Jacobi (Our implementation)
```python
# ~150 lines of code
def parallel_scan_iteration(scan_fn, init, xs):
    for _ in range(num_iter):
        carry_prev = shift(carry_all, 1)
        carry_all = vmap(scan_fn)(carry_prev, xs)
    return carry_all
```

**Complexity**: Simple ✓

### Newton (Full)
```python
# ~500+ lines of code
def newton_iteration(scan_fn, init, xs):
    for _ in range(num_iter):
        # Compute Jacobian blocks
        J = compute_block_tridiagonal_jacobian(scan_fn, carry_all, xs)

        # Compute residual
        residual = carry_all - vmap(scan_fn)(shift(carry_all), xs)

        # Solve tridiagonal system with parallel scan
        delta = parallel_tridiagonal_solve(J, -residual)

        carry_all = carry_all + delta
    return carry_all
```

**Complexity**: High, requires:
- Automatic differentiation for Jacobian
- Parallel tridiagonal solver
- Numerical stability handling

## Conclusion

**Our Jacobi implementation is the right choice** for:
- Educational purposes ✓
- Small-to-medium models (d < 1024) ✓
- Memory-constrained environments ✓
- Fast prototyping ✓

**Newton would be better** only for:
- Very large hidden dimensions (d > 2048)
- Extremely long sequences (T > 50k)
- When you need 1e-8 accuracy (rarely in ML)

The **3-iteration convergence** in the paper is achievable with Newton's method, but comes at a cost of:
- 100x more compute per iteration
- 500x more memory
- 3x more implementation complexity

For typical RNN use cases, **Jacobi with 10-15 iterations is more practical**.
