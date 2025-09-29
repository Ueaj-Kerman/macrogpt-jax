# Repository Guidelines

## Project Structure & Module Organization
The `ueaj/` package is the core library: `model/` defines configurable LLaMA variants, `train/` contains the training loop, logging, and optimizer setup, `opt/` hosts experimental optimizer kernels, and `utils/` offers configuration and compilation helpers. Dataset streaming and token preparation live in `ueaj/data/`. The `scripts/` directory provides shell helpers for running Python within the preconfigured JAX virtual environment. Tests are collected under `test/`, while `test.py` is a quick smoke example for the `config` helper.

## Build, Test, and Development Commands
- `scripts/run_python.sh python -m ueaj.train.train` — launch the distributed pretraining loop with the repository-managed interpreter; set `RUN_NAME` and `MODEL_PATH` to control logging and checkpoints.
- `scripts/run_python.sh python test.py` — verify that the configuration override system still behaves as expected.
- `python -m pytest test -q` — execute the unroll and optimizer regression tests; add `-k <pattern>` to focus on a failing scenario (e.g., `-k unroll_real`).

## Coding Style & Naming Conventions
Code is Python 3.11+, with 4-space indentation and type hints for public APIs (`ueaj/utils/configurator.py`, `ueaj/train/training_utils.py`). Use `snake_case` for functions and modules, `CamelCase` for classes, and `UPPER_SNAKE_CASE` for constants. Prefer composition via the `@config` decorator so overrides remain declarative, and keep module docstrings concise but descriptive.

## Testing Guidelines
Pytest drives regression coverage; keep new tests near related modules inside `test/` and mirror existing filenames (`test_unroll_*`). Aim to exercise both compilation-time and runtime metrics when adding optimizer features, and include asserts plus printed diagnostics only when they help trace regressions. Before opening a PR, run `python -m pytest test -q` and capture any failing seeds.

## Commit & Pull Request Guidelines
Recent history favors short, imperative summaries (e.g., `prefetch docs`, `distributed`). Group related changes per commit and note major configuration shifts in the body. Pull requests should link the relevant research ticket or issue, outline the experiment intent, and attach key artifacts: pytest results, WANDB run URLs, and, when training behavior changes, a brief loss/throughput comparison. Screenshots are only needed for visualization utilities.

## Security & Configuration Tips
Avoid hardcoding secrets: `scripts/hf_token.py` is a placeholder. Store Hugging Face tokens via `huggingface-cli login` or environment variables, and never commit real credentials. Confirm JAX cache directories (`$JAX_COMPILATION_CACHE_DIR`) exist on shared systems before running long jobs.
