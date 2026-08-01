from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from assurance_v4 import (
    _repair_mixture,
    calibration_report,
    replicate_summary,
    result_for_storage,
    simulate_manufacturing_variation,
)
from closed_loop import (
    create_recommendation_batch,
    ensure_v04_config,
    qualification_progress,
)
from project_store import ProjectStore


ROOT = Path(__file__).resolve().parents[1]


def config_for(data: pd.DataFrame) -> dict:
    mixture_columns = [c for c in data.columns if c.startswith("ingredient_")]
    baseline = data[data["status"] == "completed"].iloc[0][
        [*mixture_columns, "mix_temperature_c", "mix_time_min", "supplier_family"]
    ].to_dict()
    return ensure_v04_config({
        "mixture_columns": mixture_columns,
        "process_columns": ["mix_temperature_c", "mix_time_min"],
        "categorical_columns": ["supplier_family"],
        "response_specs": [
            {"response": "adhesion", "minimum": 8.0, "maximum": None, "weight": 1.0},
            {"response": "viscosity_cp", "minimum": 2000.0, "maximum": 2600.0, "weight": 1.0},
            {"response": "dry_time_min", "minimum": None, "maximum": 40.0, "weight": 1.0},
            {"response": "gloss", "minimum": 85.0, "maximum": None, "weight": 1.0},
        ],
        "mixture_bounds": {
            "ingredient_resin": [34.0, 62.0],
            "ingredient_crosslinker": [8.0, 20.0],
            "ingredient_solvent": [12.0, 35.0],
            "ingredient_legacy_plasticizer": [0.0, 0.0],
            "ingredient_substitute_a": [0.0, 18.0],
            "ingredient_substitute_b": [0.0, 15.0],
        },
        "process_bounds": {"mix_temperature_c": [48.0, 72.0], "mix_time_min": [18.0, 42.0]},
        "category_values": {"supplier_family": ["A", "B", "C"]},
        "mixture_total": 100.0,
        "ingredient_to_remove": "ingredient_legacy_plasticizer",
        "baseline": baseline,
        "ingredient_costs": {c: 1.0 for c in mixture_columns},
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 700,
        "min_distance": 0.05,
    })


class V04UnitTests(unittest.TestCase):
    def test_repair_mixture(self):
        x = _repair_mixture(
            np.array([50.0, 25.0, 20.0]),
            np.array([30.0, 10.0, 10.0]),
            np.array([60.0, 40.0, 40.0]),
            100.0,
        )
        self.assertAlmostEqual(float(x.sum()), 100.0, places=6)
        self.assertTrue(np.all(x >= np.array([30.0, 10.0, 10.0])))
        self.assertTrue(np.all(x <= np.array([60.0, 40.0, 40.0])))

    def test_default_config_has_explicit_gates(self):
        data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        config = config_for(data)
        self.assertEqual(set(config["qualification_gates"]), {
            "discovery", "confirmation", "process_window", "raw_material", "stability", "pilot"
        })
        self.assertEqual(config["qualification_gates"]["confirmation"]["required_replicate_groups"], 1)


class V04LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.store = ProjectStore(Path(cls.tmp.name) / "test.db")
        cls.data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        cls.config = config_for(cls.data)
        cls.project_id = cls.store.create_project("Test", cls.config)
        cls.store.import_history(cls.project_id, cls.data)
        cls.result, cls.batch_id = create_recommendation_batch(cls.store, cls.project_id, random_state=7)
        cls.store.approve_batch(cls.batch_id)
        experiments = cls.store.list_experiments(cls.project_id, batch_id=cls.batch_id)
        cls.original = experiments.iloc[0]
        cls.created_replicates = cls.store.create_replicates(str(cls.original["id"]), 2)
        valid = {"adhesion": 8.5, "viscosity_cp": 2300.0, "dry_time_min": 35.0, "gloss": 88.0}
        group = cls.store.list_experiments(cls.project_id, batch_id=cls.batch_id)
        group = group[group["replicate_group"] == cls.original["replicate_group"]]
        for _, row in group.iterrows():
            adjusted = dict(valid)
            adjusted["adhesion"] += 0.01 * int(row["replicate_index"])
            cls.store.update_experiment(
                str(row["id"]),
                status="completed",
                responses=adjusted,
                qualification_stage="confirmation",
            )
        candidate_row = cls.store.list_experiments(cls.project_id, batch_id=cls.batch_id).iloc[3]
        candidate = candidate_row.to_dict()
        candidate.update(candidate_row["recommendation"])
        robustness = simulate_manufacturing_variation(
            cls.store.project_dataframe(cls.project_id),
            config=cls.config,
            candidate=candidate,
            variation_std={"mix_temperature_c": 1.0, "mix_time_min": 1.0},
            n_simulations=100,
            random_state=5,
        )
        cls.store.save_robustness_run(
            cls.project_id, str(candidate_row["id"]), simulation_count=100,
            variation=robustness["variation_std"], result=result_for_storage(robustness),
        )
        cls.robustness = robustness

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_batch_recommendations_are_valid_mixtures(self):
        experiments = self.store.list_experiments(self.project_id, batch_id=self.batch_id)
        mixture = self.config["mixture_columns"]
        sums = experiments[mixture].astype(float).sum(axis=1)
        self.assertTrue(np.allclose(sums, 100.0, atol=1e-6))
        self.assertTrue((experiments["ingredient_legacy_plasticizer"].astype(float) == 0).all())
        distances = [float(item.get("distance_from_baseline", 0.0)) for item in experiments["recommendation"]]
        self.assertTrue(all(np.isfinite(distances)))
        self.assertLess(max(distances), 5.0)

    def test_linked_replicates_and_repeatability(self):
        self.assertEqual(len(self.created_replicates), 2)
        group = self.store.list_experiments(self.project_id, batch_id=self.batch_id)
        group = group[group["replicate_group"] == self.original["replicate_group"]]
        self.assertEqual(len(group), 3)
        summary = replicate_summary(self.store, self.project_id)
        row = summary[(summary["qualification_stage"] == "confirmation") & (summary["replicate_group"] == self.original["replicate_group"])].iloc[0]
        self.assertEqual(int(row["completed_replicates"]), 3)
        self.assertLess(float(row["cv_adhesion"]), 0.01)
        progress = qualification_progress(self.store, self.project_id)
        confirmation = progress["stage_progress"][progress["stage_progress"]["stage"] == "confirmation"].iloc[0]
        self.assertTrue(bool(confirmation["gate_passed"]))

    def test_calibration_uses_frozen_predictions(self):
        report = calibration_report(self.store, self.project_id)
        self.assertFalse(report["observations"].empty)
        self.assertFalse(report["response_summary"].empty)
        self.assertIsNotNone(report["brier_score"])
        self.assertTrue(((report["observations"]["inside_90_interval"] == True) | (report["observations"]["inside_90_interval"] == False)).all())

    def test_manufacturing_variation_and_persistence(self):
        result = self.robustness
        self.assertGreaterEqual(result["robust_success_probability"], 0.0)
        self.assertLessEqual(result["robust_success_probability"], 1.0)
        self.assertEqual(len(result["response_summary"]), 4)
        runs = self.store.list_robustness_runs(self.project_id)
        self.assertEqual(len(runs), 1)
        self.assertIn("robust_success_probability", runs.iloc[0]["result"])

    def test_audit_includes_v04_events(self):
        audit = self.store.audit_log(self.project_id)
        event_types = set(audit["event_type"])
        self.assertIn("replicates_created", event_types)
        self.assertIn("robustness_run_saved", event_types)


if __name__ == "__main__":
    unittest.main()
