# Repository Guidelines

## Project Structure & Module Organization
- `ueaj/model/` — configurable LLaMA variants.
- `ueaj/train/` — training loop, logging, optimizer setup.
- `ueaj/opt/` — experimental optimizer kernels.
- `ueaj/utils/` — configuration and compilation helpers.
- `ueaj/data/` — dataset streaming and token preparation.
- `scripts/` — shell helpers for the JAX-managed Python environment.
- `test/`, `test.py` — regression tests and a config smoke example.

## Build, Test, and Development Commands
- `scripts/run_python.sh python -m ueaj.train.train` — launch distributed pretraining; set `RUN_NAME` and `MODEL_PATH` for logging and checkpoints.
- `scripts/run_python.sh python test.py` — verify configuration override behavior.
- `python -m pytest test -q` — run unroll and optimizer regression tests; add `-k <pattern>` to focus (e.g., `-k unroll_real`).

## Coding Style & Naming Conventions
- Python 3.11+, tabs only. Use type hints for public APIs (see `ueaj/utils/configurator.py`, `ueaj/train/training_utils.py`).
- Naming: `snake_case` for functions/modules, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer composition via the `@config` decorator; keep module docstrings concise and descriptive.

## Testing Guidelines
- Pytest drives regression coverage; keep new tests near related modules inside `test/` and mirror existing filenames (e.g., `test_unroll_*`).
- Exercise both compilation-time and runtime metrics when adding optimizer features; include asserts and minimal diagnostics when they help trace regressions.
- Run locally with `python -m pytest test -q`; capture failing seeds and note them in PRs.

## Commit & Pull Request Guidelines
- Commits: short, imperative summaries (e.g., `prefetch docs`, `distributed`). Group related changes; note major configuration shifts in the body.
- PRs: link the research ticket/issue, outline experiment intent, and attach key artifacts—pytest results, WANDB run URLs, and for training behavior changes, a brief loss/throughput comparison. Screenshots only for visualization utilities.

## Security & Configuration Tips
- Do not hardcode secrets. `scripts/hf_token.py` is a placeholder—use `huggingface-cli login` or environment variables.
- Confirm `$JAX_COMPILATION_CACHE_DIR` exists on shared systems before long jobs to avoid recompilation overhead.

## Agent-Specific Instructions
- This file applies repository-wide; more-nested `AGENTS.md` files may override within their subtree.
- When modifying code, keep changes minimal and focused; avoid unrelated refactors and never commit real credentials.

