import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "NA": Path("PC_SAFT_NA_ML_Dataset.json"),
    "SA": Path("PC_SAFT_SA_ML_Dataset.json"),
}

OUTPUT_DIR = Path("audit")
OUTPUT_DIR.mkdir(exist_ok=True)

HEAVY_TAIL_VARS = ["pc_bar", "molcecular_weight", "epsilonk_k"]
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99

RANGE_RULES = {
    "molcecular_weight": (0.0, None),
    "tc_k": (0.0, None),
    "pc_bar": (0.0, None),
    "omega": (-2.0, 2.0),
    "m": (0.0, None),
    "sigma": (0.0, None),
    "epsilonk_k": (0.0, None),
    "epsilonAB_k": (0.0, None),
    "kappaAB": (0.0, 1.0),
}


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quantile(values, q):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def collect_numeric_columns(rows):
    cols = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            x = parse_number(v)
            if x is not None:
                cols[k].append(x)
    return cols


def build_hhb_imputer(all_rows):
    by_class_family = defaultdict(list)
    by_class = defaultdict(list)

    for row in all_rows:
        value = parse_number(row.get("hhb_cosm_kj_kmol"))
        if value is None:
            continue
        assoc_class = row.get("association_classification", "UNKNOWN")
        chem_family = row.get("dippr_chemical_family", "UNKNOWN")
        by_class_family[(assoc_class, chem_family)].append(value)
        by_class[assoc_class].append(value)

    family_medians = {
        key: statistics.median(vals)
        for key, vals in by_class_family.items()
        if len(vals) >= 3
    }
    class_medians = {key: statistics.median(vals) for key, vals in by_class.items() if vals}
    global_median = statistics.median([v for vals in by_class.values() for v in vals])

    def impute(row):
        assoc_class = row.get("association_classification", "UNKNOWN")
        chem_family = row.get("dippr_chemical_family", "UNKNOWN")
        family_key = (assoc_class, chem_family)

        if family_key in family_medians:
            return family_medians[family_key], "HHB_IMPUTE_CLASS_FAMILY_MEDIAN"
        if assoc_class in class_medians:
            return class_medians[assoc_class], "HHB_IMPUTE_CLASS_MEDIAN"
        return global_median, "HHB_IMPUTE_GLOBAL_MEDIAN"

    return impute


def detect_impossible_ranges(rows, dataset_name):
    issues = []
    for idx, row in enumerate(rows):
        for col, (low, high) in RANGE_RULES.items():
            if col not in row:
                continue
            x = parse_number(row.get(col))
            if x is None:
                continue
            if low is not None and x < low:
                issues.append((dataset_name, idx, row, col, x, f"<{low}"))
            elif high is not None and x > high:
                issues.append((dataset_name, idx, row, col, x, f">{high}"))
    return issues


def apply_heavy_tail_policy(rows, policy):
    for row in rows:
        for col, params in policy.items():
            x = parse_number(row.get(col))
            if x is None:
                continue
            winsorized = min(max(x, params["winsor_low"]), params["winsor_high"])
            scale = params["iqr"] if params["iqr"] != 0 else 1.0
            row[f"{col}_winsorized"] = round(winsorized, 8)
            row[f"{col}_robust_scaled"] = round((winsorized - params["median"]) / scale, 8)


def main():
    datasets = {name: json.loads(path.read_text()) for name, path in DATASETS.items()}
    all_rows = [row for rows in datasets.values() for row in rows]

    hhb_imputer = build_hhb_imputer(all_rows)

    numeric_all = collect_numeric_columns(all_rows)
    heavy_tail_policy = {}
    for col in HEAVY_TAIL_VARS:
        vals = numeric_all.get(col, [])
        if not vals:
            continue
        q01 = quantile(vals, WINSOR_Q_LOW)
        q25 = quantile(vals, 0.25)
        q50 = quantile(vals, 0.50)
        q75 = quantile(vals, 0.75)
        q99 = quantile(vals, WINSOR_Q_HIGH)
        heavy_tail_policy[col] = {
            "winsor_low": q01,
            "winsor_high": q99,
            "median": q50,
            "iqr": q75 - q25,
        }

    audit_rows = []

    for name, rows in datasets.items():
        impossible = detect_impossible_ranges(rows, name)
        for issue in impossible:
            _, idx, row, col, value, reason = issue
            audit_rows.append({
                "dataset": name,
                "row_index": idx,
                "compound": row.get("name_dippr"),
                "casrn": row.get("casrn"),
                "feature": col,
                "issue_type": "IMPOSSIBLE_RANGE",
                "issue_detail": f"value {value} outside expected range ({reason})",
                "action": "FLAG_ONLY",
                "action_reason_code": "RANGE_CHECK_FAILED",
            })

        numeric_columns = collect_numeric_columns(rows)
        for col, vals in numeric_columns.items():
            q01 = quantile(vals, 0.01)
            q99 = quantile(vals, 0.99)
            for idx, row in enumerate(rows):
                x = parse_number(row.get(col))
                if x is None:
                    continue
                if q01 is not None and x < q01:
                    audit_rows.append({
                        "dataset": name,
                        "row_index": idx,
                        "compound": row.get("name_dippr"),
                        "casrn": row.get("casrn"),
                        "feature": col,
                        "issue_type": "EXTREME_QUANTILE",
                        "issue_detail": f"value {x} < q01 {q01}",
                        "action": "WINSORIZE_IF_HEAVY_TAIL",
                        "action_reason_code": "QUANTILE_LOW_OUTLIER",
                    })
                elif q99 is not None and x > q99:
                    audit_rows.append({
                        "dataset": name,
                        "row_index": idx,
                        "compound": row.get("name_dippr"),
                        "casrn": row.get("casrn"),
                        "feature": col,
                        "issue_type": "EXTREME_QUANTILE",
                        "issue_detail": f"value {x} > q99 {q99}",
                        "action": "WINSORIZE_IF_HEAVY_TAIL",
                        "action_reason_code": "QUANTILE_HIGH_OUTLIER",
                    })

        for idx, row in enumerate(rows):
            if row.get("hhb_cosm_kj_kmol") is None:
                imputed, reason_code = hhb_imputer(row)
                row["hhb_cosm_kj_kmol"] = round(imputed, 8)
                audit_rows.append({
                    "dataset": name,
                    "row_index": idx,
                    "compound": row.get("name_dippr"),
                    "casrn": row.get("casrn"),
                    "feature": "hhb_cosm_kj_kmol",
                    "issue_type": "NULL",
                    "issue_detail": "null value",
                    "action": f"IMPUTED_TO_{row['hhb_cosm_kj_kmol']}",
                    "action_reason_code": reason_code,
                })

        apply_heavy_tail_policy(rows, heavy_tail_policy)

    # Write version-controlled transformed datasets.
    for name, rows in datasets.items():
        output_path = OUTPUT_DIR / f"PC_SAFT_{name}_ML_Dataset_preprocessed.json"
        output_path.write_text(json.dumps(rows, indent=2))

    # Write audit CSV.
    audit_csv_path = OUTPUT_DIR / "data_audit_report.csv"
    fieldnames = [
        "dataset",
        "row_index",
        "compound",
        "casrn",
        "feature",
        "issue_type",
        "issue_detail",
        "action",
        "action_reason_code",
    ]
    with audit_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    # Write markdown summary.
    md_path = OUTPUT_DIR / "data_audit_report.md"
    summary = defaultdict(int)
    for row in audit_rows:
        summary[(row["issue_type"], row["action_reason_code"])] += 1

    lines = [
        "# Data Audit Report",
        "",
        "## Scope",
        "- Datasets: `PC_SAFT_NA_ML_Dataset.json`, `PC_SAFT_SA_ML_Dataset.json`.",
        "- Checks: nulls, impossible ranges, extreme q01/q99 quantiles.",
        "- Deterministic null policy for `hhb_cosm_kj_kmol`: class+family median (n>=3), else class median, else global median.",
        f"- Heavy-tail policy (applied to all scenarios): winsorize to [{WINSOR_Q_LOW:.2f}, {WINSOR_Q_HIGH:.2f}] then robust scale by median/IQR.",
        "",
        "## Heavy-Tail Policy Parameters",
        "| Feature | Winsor Low (q01) | Winsor High (q99) | Median | IQR |",
        "|---|---:|---:|---:|---:|",
    ]
    for col, params in heavy_tail_policy.items():
        lines.append(
            f"| {col} | {params['winsor_low']:.6f} | {params['winsor_high']:.6f} | {params['median']:.6f} | {params['iqr']:.6f} |"
        )

    lines.extend([
        "",
        "## Audit Findings Summary",
        "| Issue Type | Reason Code | Count |",
        "|---|---|---:|",
    ])
    for (issue, reason), count in sorted(summary.items()):
        lines.append(f"| {issue} | {reason} | {count} |")

    range_failures = [r for r in audit_rows if r["issue_type"] == "IMPOSSIBLE_RANGE"]
    lines.extend([
        "",
        "## Impossible Range Findings",
        f"- Count: {len(range_failures)}",
    ])

    hhb_actions = [r for r in audit_rows if r["feature"] == "hhb_cosm_kj_kmol" and r["issue_type"] == "NULL"]
    lines.extend([
        "",
        "## `hhb_cosm_kj_kmol` Null Handling (Affected Compounds)",
        "| Dataset | Compound | CASRN | Action | Reason Code |",
        "|---|---|---|---|---|",
    ])
    for row in hhb_actions:
        lines.append(
            f"| {row['dataset']} | {row['compound']} | {row['casrn']} | {row['action']} | {row['action_reason_code']} |"
        )

    lines.extend([
        "",
        "## Version-Controlled Outputs",
        "- `audit/data_audit_report.csv`",
        "- `audit/data_audit_report.md`",
        "- `audit/PC_SAFT_NA_ML_Dataset_preprocessed.json`",
        "- `audit/PC_SAFT_SA_ML_Dataset_preprocessed.json`",
    ])

    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
