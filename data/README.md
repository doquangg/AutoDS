## `data/`

Input datasets and benchmark configuration.

- **`sample_data/healthcare/`** — small dirty healthcare CSV used for ad-hoc runs and manual smoke testing.
- **`benchmark_data/`** — UCI datasets downloaded by `scripts/download_benchmarks.py` (requires `ucimlrepo`). Consumed by `scripts/evaluate_benchmarks.py`.
- **`benchmark_metadata.yaml`** — per-dataset configuration for the benchmark harness: target column, task type, published baseline, and any dataset-specific overrides.
