from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from analysis.common import ensure_output_dirs, load_config, load_dataset


def main(config_path: str) -> None:
    config = load_config(config_path)
    ensure_output_dirs(config)

    sa = load_dataset(config["datasets"]["self_associating"])
    family_col = config["columns"]["family"]
    scheme_col = config["columns"]["association_scheme"]

    rule_df = sa[[family_col, "Area_HA", "Area_HD", scheme_col, config["columns"]["id"]]].dropna().copy()

    family_mode = rule_df.groupby(family_col)[scheme_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])

    def predict(row):
        if row["Area_HA"] > 0 and row["Area_HD"] > 0:
            return "2B"
        if row["Area_HA"] > 0 and row["Area_HD"] == 0:
            return "1A"
        if row["Area_HD"] > 0 and row["Area_HA"] == 0:
            return "1B"
        return family_mode.get(row[family_col], "2B")

    rule_df["pred_association_scheme"] = rule_df.apply(predict, axis=1)

    acc = accuracy_score(rule_df[scheme_col], rule_df["pred_association_scheme"])
    metrics = pd.DataFrame(
        [
            {
                "dataset": "SA",
                "scenario": "association_rule",
                "model": "rule_based",
                "target": scheme_col,
                "accuracy": acc,
                "n_samples": len(rule_df),
                "status": "ok",
            }
        ]
    )

    metrics.to_csv(Path(config["outputs"]["validation_dir"]) / "association_rule_metrics.csv", index=False)
    rule_df.to_csv(Path(config["outputs"]["validation_dir"]) / "association_rule_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_settings.json")
    args = parser.parse_args()
    main(args.config)
