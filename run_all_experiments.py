from __future__ import annotations

import argparse
import subprocess
import sys

STAGES = [
    "analysis/eda/generate_eda_summary.py",
    "analysis/models/train_scenarios.py",
    "analysis/models/transfer_learning_sa.py",
    "analysis/validation/association_scheme_rule.py",
    "analysis/validation/property_deviation.py",
]


def main(config_path: str) -> None:
    for stage in STAGES:
        cmd = [sys.executable, stage, "--config", config_path]
        print(f"[pipeline] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full PC-SAFT ML experiment pipeline")
    parser.add_argument("--config", default="configs/experiment_settings.json")
    args = parser.parse_args()
    main(args.config)
