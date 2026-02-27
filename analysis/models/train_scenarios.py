from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.common import ensure_output_dirs, load_config, load_dataset

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def make_preprocessor(features: List[str], frame: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical = [c for c in features if frame[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return preprocessor, numeric, categorical


def build_model(model_name: str, params: Dict, seed: int):
    if model_name == "random_forest":
        return MultiOutputRegressor(RandomForestRegressor(random_state=seed, **params))
    if model_name == "mlp":
        return MultiOutputRegressor(MLPRegressor(random_state=seed, **params))
    if model_name == "xgboost":
        if not HAS_XGBOOST:
            return None
        model = XGBRegressor(random_state=seed, objective="reg:squarederror", **params)
        return MultiOutputRegressor(model)
    raise ValueError(model_name)


def main(config_path: str) -> None:
    config = load_config(config_path)
    ensure_output_dirs(config)

    df = load_dataset(config["datasets"]["non_associating"])
    targets = config["targets"]["primary"]
    id_col = config["columns"]["id"]
    family_col = config["columns"]["family"]

    metrics_rows = []
    predictions = []

    for scenario in ["scenario_2a", "scenario_2b", "scenario_2c"]:
        features = config["features"][scenario]
        work = df[features + targets + [id_col, family_col]].dropna().copy()

        X = work[features]
        y = work[targets]
        strat = work[family_col] if family_col in work.columns else None

        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X,
            y,
            work[[id_col, family_col]],
            test_size=config["splits"]["test_size"],
            random_state=config["random_seeds"]["split"],
            stratify=strat,
        )

        preprocessor, _, _ = make_preprocessor(features, work)

        for model_name in ["random_forest", "xgboost", "mlp"]:
            model = build_model(model_name, config["models"][model_name], config["random_seeds"]["global"])
            if model is None:
                metrics_rows.append(
                    {
                        "dataset": "NA",
                        "scenario": scenario,
                        "model": model_name,
                        "target": "ALL",
                        "status": "skipped",
                        "reason": "xgboost_not_installed",
                    }
                )
                continue

            pipe = Pipeline([("pre", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            for i, target in enumerate(targets):
                metrics_rows.append(
                    {
                        "dataset": "NA",
                        "scenario": scenario,
                        "model": model_name,
                        "target": target,
                        "mae": mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
                        "r2": r2_score(y_test.iloc[:, i], y_pred[:, i]),
                        "n_test": len(y_test),
                        "status": "ok",
                    }
                )

            pred_df = pd.DataFrame(y_pred, columns=[f"pred_{t}" for t in targets])
            truth_df = y_test.reset_index(drop=True).rename(columns={t: f"true_{t}" for t in targets})
            meta_df = meta_test.reset_index(drop=True)
            merged = pd.concat([meta_df, truth_df, pred_df], axis=1)
            merged["dataset"] = "NA"
            merged["scenario"] = scenario
            merged["model"] = model_name
            predictions.append(merged)

    metrics_path = Path(config["outputs"]["metrics_dir"]) / "scenario_training_metrics.csv"
    preds_path = Path(config["outputs"]["predictions_dir"]) / "scenario_training_predictions.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(preds_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_settings.json")
    args = parser.parse_args()
    main(args.config)
