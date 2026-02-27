from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.common import ensure_output_dirs, load_config, load_dataset


def summarize_dataset(df: pd.DataFrame, family_col: str, dataset_tag: str, outdir: Path) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    summary = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})
    summary["dataset"] = dataset_tag
    summary.to_csv(outdir / f"{dataset_tag}_numeric_summary.csv", index=False)

    family_counts = (
        df.groupby([family_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    family_counts["dataset"] = dataset_tag
    family_counts.to_csv(outdir / f"{dataset_tag}_family_stratification.csv", index=False)


def main(config_path: str) -> None:
    config = load_config(config_path)
    ensure_output_dirs(config)
    outdir = Path(config["outputs"]["eda_dir"])
    family_col = config["columns"]["family"]

    na = load_dataset(config["datasets"]["non_associating"])
    sa = load_dataset(config["datasets"]["self_associating"])

    summarize_dataset(na, family_col, "NA", outdir)
    summarize_dataset(sa, family_col, "SA", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_settings.json")
    args = parser.parse_args()
    main(args.config)
