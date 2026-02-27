#!/usr/bin/env python3
"""Validate that all equation IDs in DIPPR_Master.json are implemented in DIPPREvaluator."""

import ast
import json
import re
import sys
from pathlib import Path


def iter_equation_refs(master_data):
    for compound in master_data:
        chemid = compound.get("chemid")
        correlations = compound.get("correlations", {})
        for prop, corr in correlations.items():
            yield chemid, prop, int(corr.get("equation_id"))


def parse_supported_ids(evaluator_path: Path):
    source = evaluator_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    supported = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            match = re.fullmatch(r"eq(\d+)", node.name)
            if match:
                supported.add(int(match.group(1)))

    return supported


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    master_path = repo_root / "DIPPR_Master.json"
    evaluator_path = repo_root / "dippr_evaluator.py"

    with master_path.open("r", encoding="utf-8") as fh:
        master_data = json.load(fh)

    supported = parse_supported_ids(evaluator_path)
    missing = {}

    for chemid, prop, eq_id in iter_equation_refs(master_data):
        if eq_id not in supported:
            missing.setdefault(eq_id, []).append((chemid, prop))

    print(f"Supported equation IDs in DIPPREvaluator: {sorted(supported)}")

    if missing:
        print("\nERROR: Unsupported equation IDs found in DIPPR_Master.json:")
        for eq_id in sorted(missing):
            refs = ", ".join(f"(chemid={c}, property={p})" for c, p in missing[eq_id][:5])
            extra = "" if len(missing[eq_id]) <= 5 else f" ... +{len(missing[eq_id]) - 5} more"
            print(f"  - eq_id={eq_id}: {refs}{extra}")
        return 1

    print("All equation IDs in DIPPR_Master.json are implemented in DIPPREvaluator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
