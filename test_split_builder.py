import unittest

from split_builder import (
    build_compound_registry,
    build_experiment_splits,
    load_json,
    NA_DATASET_PATH,
    SA_DATASET_PATH,
)


class SplitBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sa_rows = load_json(SA_DATASET_PATH)
        na_rows = load_json(NA_DATASET_PATH)
        cls.registry = build_compound_registry(sa_rows, na_rows)
        cls.splits = build_experiment_splits(cls.registry)

    def test_unique_ids_single_split_per_experiment(self):
        for experiment, split_data in self.splits.items():
            train = set(split_data["train"])
            val = set(split_data["val"])
            test = set(split_data["test"])
            self.assertFalse(train & val, f"{experiment} has train-val leakage")
            self.assertFalse(train & test, f"{experiment} has train-test leakage")
            self.assertFalse(val & test, f"{experiment} has val-test leakage")

    def test_transfer_training_isolated_from_universal_validation(self):
        universal_val = set(self.splits["universal"]["val"])
        transfer_train = set(self.splits["transfer_epsilonAB_k_kappaAB"]["train"])
        self.assertFalse(
            transfer_train & universal_val,
            "Transfer train IDs overlap universal validation IDs",
        )


if __name__ == "__main__":
    unittest.main()
