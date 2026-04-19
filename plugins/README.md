## `plugins/`

Non-LLM components that the pipeline calls into.

- **`profiler.py`** — data profiling. Computes per-column stats (dtype, missingness, cardinality, samples), runs `ydata-profiling` in detailed mode, and produces a structured `DatasetProfile` plus quality scores that drive the investigator's decisions and the multi-pass termination check. Also exposes a fast `detailed_profiler=False` mode used by the FE re-profile loop.

- **`modeller.py`** — AutoGluon training wrapper. Configurable via `AUTOGLUON_TIME_LIMIT` (default `300`s) and `AUTOGLUON_PRESETS` (default `medium_quality`). Returns the fitted predictor plus metadata consumed by the answer agent. Supports an explicit validation fold, used by the benchmark harness.
