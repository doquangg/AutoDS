# scripts/download_benchmarks.py
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

Path("data/benchmark_data").mkdir(parents=True, exist_ok=True)


def save_dataset(ds, path):
    parts = [p for p in [ds.data.features, ds.data.targets] if p is not None]
    if not parts:
        print(f"Skipping {path}: no data returned")
        return
    pd.concat(parts, axis=1).to_parquet(path)


# Diabetes 130-US Hospitals (UCI #296)
save_dataset(fetch_ucirepo(id=296), "data/benchmark_data/diabetes_readmission.parquet")

# SECOM (UCI #601)
save_dataset(fetch_ucirepo(id=601), "data/benchmark_data/secom.parquet")

# Taiwan Credit Card Default (UCI #350)
save_dataset(fetch_ucirepo(id=350), "data/benchmark_data/taiwan_credit_default.parquet")
