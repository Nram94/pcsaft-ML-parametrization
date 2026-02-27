import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

SA_DATASET_PATH = Path("PC_SAFT_SA_ML_Dataset.json")
NA_DATASET_PATH = Path("PC_SAFT_NA_ML_Dataset.json")
MANIFEST_DIR = Path("split_manifests")


def normalize_casrn(casrn: str) -> str:
    """Normalize CASRN values stored with extra quotes in source JSON."""
    return str(casrn).strip().strip('"')


def compound_id(entry: Dict) -> str:
    """Prefer DIPPR CHEMID and fallback to CASRN when CHEMID is not available."""
    chemid = entry.get("dippr_chemid")
    if chemid not in (None, "", "null"):
        return f"dippr:{int(chemid)}"

    casrn = entry.get("casrn")
    if casrn not in (None, "", "null"):
        return f"cas:{normalize_casrn(casrn)}"

    raise ValueError("Entry is missing both dippr_chemid and casrn.")


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_compound_registry(sa_rows: List[Dict], na_rows: List[Dict]) -> Dict[str, Dict]:
    registry: Dict[str, Dict] = {}

    def _visit(rows: Iterable[Dict], source: str) -> None:
        for row in rows:
            cid = compound_id(row)
            if cid not in registry:
                registry[cid] = {
                    "compound_id": cid,
                    "dippr_chemid": row.get("dippr_chemid"),
                    "casrn": normalize_casrn(row.get("casrn", "")) if row.get("casrn") else None,
                    "name_dippr": row.get("name_dippr"),
                    "sources": set(),
                    "has_epsilonAB_k": "epsilonAB_k" in row and row.get("epsilonAB_k") is not None,
                    "has_kappaAB": "kappaAB" in row and row.get("kappaAB") is not None,
                }
            registry[cid]["sources"].add(source)

    _visit(sa_rows, "SA")
    _visit(na_rows, "NA")

    for info in registry.values():
        sources = info["sources"]
        if sources == {"SA", "NA"}:
            overlap = "overlap"
        elif sources == {"SA"}:
            overlap = "sa_only"
        else:
            overlap = "na_only"
        info["overlap_status"] = overlap
        info["sources"] = sorted(sources)

    return registry


def split_ids(unique_ids: Iterable[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    ids = list(set(unique_ids))
    random.Random(seed).shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def assert_no_leakage(splits: Dict[str, List[str]], experiment_name: str) -> None:
    train_ids, val_ids, test_ids = map(set, (splits["train"], splits["val"], splits["test"]))

    overlaps: List[Tuple[str, Set[str]]] = []
    if train_ids & val_ids:
        overlaps.append(("train-val", train_ids & val_ids))
    if train_ids & test_ids:
        overlaps.append(("train-test", train_ids & test_ids))
    if val_ids & test_ids:
        overlaps.append(("val-test", val_ids & test_ids))

    if overlaps:
        detail = "; ".join(f"{name}={sorted(ids)}" for name, ids in overlaps)
        raise AssertionError(f"Leakage detected for '{experiment_name}': {detail}")


def build_experiment_splits(registry: Dict[str, Dict], seed: int = 42) -> Dict[str, Dict]:
    all_ids = sorted(registry)
    universal = split_ids(all_ids, train_ratio=0.7, val_ratio=0.15, seed=seed)
    assert_no_leakage(universal, "universal")

    sa_ids = [cid for cid, meta in registry.items() if "SA" in meta["sources"]]
    universal_val_ids = set(universal["val"])

    transfer_train_candidates = sorted(set(sa_ids) - universal_val_ids)
    transfer_train = split_ids(transfer_train_candidates, train_ratio=0.8, val_ratio=0.0, seed=seed + 1)["train"]

    transfer_holdout_ids = sorted(set(sa_ids) - set(transfer_train))
    transfer_holdout = split_ids(transfer_holdout_ids, train_ratio=0.5, val_ratio=0.0, seed=seed + 2)
    transfer = {
        "train": transfer_train,
        "val": transfer_holdout["train"],
        "test": transfer_holdout["test"],
        "isolation_reference_universal_val": sorted(universal_val_ids),
    }

    assert_no_leakage(transfer, "transfer_epsilonAB_k_kappaAB")

    if set(transfer["train"]) & universal_val_ids:
        raise AssertionError(
            "Transfer-learning SA training IDs overlap universal-model validation IDs."
        )

    return {
        "universal": universal,
        "transfer_epsilonAB_k_kappaAB": transfer,
    }


def write_manifests(registry: Dict[str, Dict], splits: Dict[str, Dict]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    registry_payload = {
        "total_unique_compounds": len(registry),
        "compounds": sorted(registry.values(), key=lambda x: x["compound_id"]),
    }
    with (MANIFEST_DIR / "compound_registry.json").open("w", encoding="utf-8") as fh:
        json.dump(registry_payload, fh, indent=2)

    split_payload = {
        "id_key_policy": "dippr_chemid preferred, CASRN fallback",
        "experiments": splits,
        "counts": {
            exp_name: {split_name: len(ids) for split_name, ids in exp.items() if isinstance(ids, list)}
            for exp_name, exp in splits.items()
        },
    }
    with (MANIFEST_DIR / "split_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(split_payload, fh, indent=2)


def main() -> None:
    sa_rows = load_json(SA_DATASET_PATH)
    na_rows = load_json(NA_DATASET_PATH)

    registry = build_compound_registry(sa_rows, na_rows)
    splits = build_experiment_splits(registry)
    write_manifests(registry, splits)


if __name__ == "__main__":
    main()
