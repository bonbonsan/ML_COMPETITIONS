# notebooks

Exploratory notebooks that demonstrate how to use `my_library` in interactive workflows. They are meant for quick experiments, GPU benchmarks, and feature engineering prototypes.

## Structure

- `data/` – Helper datasets or intermediate artifacts used by the notebooks.
- `GPU_comparison.ipynb` – Benchmarking experiments comparing CPU vs GPU training flows.
- `sample.ipynb` – Minimal end-to-end example showing dataset loading, training, and inference.

## Recommended Workflow

1. Launch JupyterLab via `scripts/run_jupyter-lab.sh` (inside the Docker container if GPU resources are needed).
2. Set `PYTHONPATH=.` or install the package in editable mode so notebooks can import `my_library`.
3. Keep notebooks lightweight and promote stable logic into reusable modules inside the package.
