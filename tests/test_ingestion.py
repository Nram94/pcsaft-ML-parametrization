import json
import tempfile
import unittest
from pathlib import Path

from ingestion import (
    canonicalize_dippr_record,
    canonicalize_pc_saft_record,
    preflight_validate,
    validate_file,
    PreflightValidationError,
)


class IngestionSchemaTests(unittest.TestCase):
    def test_pc_saft_alias_for_misspelled_weight(self):
        rec = {"molcecular_weight": 10.1, "tc_k": 500}
        out = canonicalize_pc_saft_record(rec)
        self.assertIn("molecular_weight", out)
        self.assertNotIn("molcecular_weight", out)
        self.assertEqual(out["molecular_weight"], 10.1)

    def test_dippr_mw_maps_to_canonical_weight(self):
        rec = {"critical_props": {"mw": 44.01, "tc": 304.2}}
        out = canonicalize_dippr_record(rec)
        self.assertEqual(out["molecular_weight"], 44.01)

    def test_preflight_catches_missing_required_feature(self):
        with self.assertRaises(PreflightValidationError):
            preflight_validate([{"tc_k": 300}], ["molecular_weight", "tc_k"])

    def test_validate_file_structural_passes_with_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.json"
            path.write_text(json.dumps([{"molcecular_weight": 18.0}]))
            count = validate_file(path, source="pc_saft", scenario="structural")
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
