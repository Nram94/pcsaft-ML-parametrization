from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.common import ensure_output_dirs, load_config, load_dataset

PROPERTY_COLUMNS = {
    "VP": "errPsat_%",
    "rhoL": "errvLsat_%",
    "CpL": "errcpLsat_%",
    "Hvap": "errΔvapH_%",
}


def summarize_deviations(df: pd.DataFrame, id_col: str, dataset: str, scenario: str, model: str) -> pd.DataFrame:
    rows = []
    for prop, col in PROPERTY_COLUMNS.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "model": model,
                "property": prop,
                "source_column": col,
                "mean_abs_dev_percent": values.abs().mean(),
                "median_abs_dev_percent": values.abs().median(),
                "n_points": values.shape[0],
            }
        )
    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    config = load_config(config_path)
    ensure_output_dirs(config)

    id_col = config["columns"]["id"]
    na = load_dataset(config["datasets"]["non_associating"])
    sa = load_dataset(config["datasets"]["self_associating"])

    tagged = []
    for dataset, frame in [("NA", na), ("SA", sa)]:
        tagged.append(summarize_deviations(frame, id_col, dataset, "raw_dataset", "pcsaft_reference"))

    pred_dir = Path(config["outputs"]["predictions_dir"])
    if pred_dir.exists():
        for pred_file in pred_dir.glob("*.csv"):
            pred = pd.read_csv(pred_file)
            if {"dataset", "scenario", "model", id_col}.issubset(pred.columns):
                for (dataset, scenario, model), sub in pred.groupby(["dataset", "scenario", "model"]):
                    base = na if dataset == "NA" else sa
                    merged = base.merge(sub[[id_col]].drop_duplicates(), on=id_col, how="inner")
                    tagged.append(summarize_deviations(merged, id_col, dataset, scenario, model))

    out = pd.concat([x for x in tagged if not x.empty], ignore_index=True)
    out.to_csv(Path(config["outputs"]["validation_dir"]) / "property_deviation_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_settings.json")
    args = parser.parse_args()
    main(args.config)
