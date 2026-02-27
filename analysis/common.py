from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requested but PyYAML is not installed. Use JSON config.") from exc
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported config format: {path}")


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    for key in ["metrics_dir", "predictions_dir", "eda_dir", "validation_dir"]:
        Path(config["outputs"][key]).mkdir(parents=True, exist_ok=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path)


def tag_frame(df: pd.DataFrame, **tags: str) -> pd.DataFrame:
    tagged = df.copy()
    for k, v in tags.items():
        tagged[k] = v
    return tagged
