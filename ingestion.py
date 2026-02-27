import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCHEMA_PATH = Path("data/schema.json")
CANONICAL_ALIASES = {
    "molcecular_weight": "molecular_weight",
}


class PreflightValidationError(ValueError):
    """Raised when required ingestion features are missing or misspelled."""


def load_schema(path: Path = SCHEMA_PATH) -> Dict[str, Dict[str, List[str]]]:
    with path.open() as f:
        return json.load(f)


def _canonicalize_aliases(record: Dict[str, Any]) -> Dict[str, Any]:
    canonical = dict(record)
    for alias, canonical_name in CANONICAL_ALIASES.items():
        if alias in canonical and canonical_name not in canonical:
            canonical[canonical_name] = canonical[alias]
        canonical.pop(alias, None)
    return canonical


def canonicalize_pc_saft_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize a PC-SAFT training record without mutating raw inputs."""
    return _canonicalize_aliases(record)


def canonicalize_dippr_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize a DIPPR record, including critical_props.mw -> molecular_weight."""
    canonical = _canonicalize_aliases(record)
    critical_props = canonical.get("critical_props") or {}
    if isinstance(critical_props, dict) and "mw" in critical_props and "molecular_weight" not in canonical:
        canonical["molecular_weight"] = critical_props["mw"]
    return canonical


def canonicalize_records(records: Iterable[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    if source == "dippr":
        return [canonicalize_dippr_record(record) for record in records]
    if source == "pc_saft":
        return [canonicalize_pc_saft_record(record) for record in records]
    raise ValueError(f"Unsupported source '{source}'. Expected one of: dippr, pc_saft")


def preflight_validate(records: Iterable[Dict[str, Any]], required_fields: Iterable[str]) -> None:
    required = list(required_fields)
    for idx, record in enumerate(records):
        missing = [field for field in required if field not in record or record[field] is None]
        if missing:
            record_keys = list(record.keys())
            suggestions = {}
            for field in missing:
                if field in record:
                    continue
                suggestion = difflib.get_close_matches(field, record_keys, n=1, cutoff=0.75)
                if suggestion and suggestion[0] != field:
                    suggestions[field] = suggestion[0]
            hint = ""
            if suggestions:
                pairs = ", ".join(f"{k}->{v}" for k, v in suggestions.items())
                hint = f" Possible misspellings: {pairs}."
            raise PreflightValidationError(
                f"Record {idx} is missing required fields: {missing}.{hint}"
            )


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return data


def validate_file(path: Path, source: str, scenario: str, schema_path: Path = SCHEMA_PATH) -> int:
    schema = load_schema(schema_path)
    scenarios = schema.get("scenarios", {})
    if scenario not in scenarios:
        raise ValueError(f"Scenario '{scenario}' is not defined in schema {schema_path}")

    records = load_json_records(path)
    canonical_records = canonicalize_records(records, source=source)
    preflight_validate(canonical_records, scenarios[scenario]["required_fields"])
    return len(canonical_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical ingestion + preflight validation")
    parser.add_argument("--path", required=True, help="JSON dataset path")
    parser.add_argument("--source", choices=["pc_saft", "dippr"], required=True)
    parser.add_argument("--scenario", choices=["structural", "macroscopic", "quantum"], required=True)
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="Schema JSON path")
    args = parser.parse_args()

    total = validate_file(Path(args.path), args.source, args.scenario, Path(args.schema))
    print(f"Preflight passed for {total} records in '{args.path}' using scenario '{args.scenario}'.")


if __name__ == "__main__":
    main()
