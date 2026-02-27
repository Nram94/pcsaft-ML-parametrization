from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.common import ensure_output_dirs, load_config, load_dataset


def main(config_path: str) -> None:
    config = load_config(config_path)
    ensure_output_dirs(config)

    na = load_dataset(config["datasets"]["non_associating"])
    sa = load_dataset(config["datasets"]["self_associating"])

    base_feats = [f for f in config["features"]["base"] if f in na.columns]
    source_target = "epsilonk_k"

    na_train = na[base_feats + [source_target]].dropna()
    source_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=config["random_seeds"]["global"], n_estimators=300)),
        ]
    )
    source_model.fit(na_train[base_feats], na_train[source_target])

    sa_work = sa[base_feats + config["targets"]["transfer_sa"] + [config["columns"]["id"]]].dropna().copy()
    sa_work["transfer_epsilonk_k"] = source_model.predict(sa_work[base_feats])

    metrics_rows = []
    preds = []

    for target in config["targets"]["transfer_sa"]:
        X = sa_work[base_feats + ["transfer_epsilonk_k"]]
        y = sa_work[target]
        meta = sa_work[[config["columns"]["id"]]].copy()
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            X,
            y,
            meta,
            test_size=config["splits"]["test_size"],
            random_state=config["random_seeds"]["split"],
        )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(random_state=config["random_seeds"]["global"], n_estimators=250)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_rows.append(
            {
                "dataset": "SA",
                "scenario": "transfer_learning_sa",
                "model": "rf_transfer",
                "target": target,
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "n_test": len(y_test),
                "status": "ok",
            }
        )

        p = m_test.reset_index(drop=True)
        p[f"true_{target}"] = y_test.reset_index(drop=True)
        p[f"pred_{target}"] = y_pred
        p["dataset"] = "SA"
        p["scenario"] = "transfer_learning_sa"
        p["model"] = "rf_transfer"
        p["target"] = target
        preds.append(p)

    pd.DataFrame(metrics_rows).to_csv(Path(config["outputs"]["metrics_dir"]) / "transfer_learning_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(
        Path(config["outputs"]["predictions_dir"]) / "transfer_learning_predictions.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_settings.json")
    args = parser.parse_args()
    main(args.config)
