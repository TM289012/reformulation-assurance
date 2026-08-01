from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from assurance_v4 import calibration_report, simulate_manufacturing_variation
from closed_loop import ensure_v04_config
from dossier import compute_evidence_hash
from pilot_store import PilotStore
from process_window import design_process_window
from project_store import ProjectStore


ROOT = Path(__file__).resolve().parents[1]


def config_for(data: pd.DataFrame) -> dict:
    mixture_columns = [c for c in data.columns if c.startswith("ingredient_")]
    completed = data[data["status"] == "completed"]
    baseline = completed.iloc[0][
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
        "candidate_pool_size": 500,
        "min_distance": 0.05,
    })


def one_recommendation(config: dict) -> pd.DataFrame:
    row = dict(config["baseline"])
    removed = float(row["ingredient_legacy_plasticizer"])
    row["ingredient_legacy_plasticizer"] = 0.0
    row["ingredient_resin"] = float(row["ingredient_resin"]) + removed
    row.update({
        "purpose": "Confirmed candidate",
        "predicted_adhesion": 8.6,
        "uncertainty_adhesion": 0.4,
        "predicted_viscosity_cp": 2350.0,
        "uncertainty_viscosity_cp": 100.0,
        "predicted_dry_time_min": 35.0,
        "uncertainty_dry_time_min": 2.0,
        "predicted_gloss": 89.0,
        "uncertainty_gloss": 1.5,
        "probability_all_specs": 0.85,
    })
    return pd.DataFrame([row])


class V062WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        self.config = config_for(self.data)
        self.store = ProjectStore(Path(self.tmp.name) / "test.db")
        self.project = self.store.create_project("v062", self.config)
        self.store.import_history(self.project, self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_process_window_design_and_stage(self):
        nominal = one_recommendation(self.config).iloc[0].to_dict()
        preview = design_process_window(
            config=self.config,
            nominal_inputs=nominal,
            source_replicate_group="B01-E01",
            process_columns=["mix_temperature_c", "mix_time_min"],
            deltas={"mix_temperature_c": 3.0, "mix_time_min": 2.0},
            mode="corners",
        )
        self.assertEqual(len(preview), 5)
        self.assertTrue((preview["ingredient_legacy_plasticizer"] == 0).all())
        self.assertEqual(len(preview[["mix_temperature_c", "mix_time_min"]].drop_duplicates()), 5)
        batch = self.store.create_batch(
            self.project,
            preview,
            decision="RUN PROCESS WINDOW",
            decision_reason="test",
            qualification_stage="process_window",
        )
        self.store.approve_batch(batch)
        experiments = self.store.list_experiments(self.project, batch_id=batch)
        self.assertTrue((experiments["qualification_stage"] == "process_window").all())
        fingerprint = preview.iloc[0]["design_fingerprint"]
        self.assertTrue(self.store.has_design_fingerprint(self.project, fingerprint))

    def test_completed_requires_nonzero_complete_measurements(self):
        batch = self.store.create_batch(
            self.project, one_recommendation(self.config), decision="RUN", decision_reason="test"
        )
        self.store.approve_batch(batch)
        experiment = self.store.list_experiments(self.project, batch_id=batch).iloc[0]
        with self.assertRaisesRegex(ValueError, "every measurement"):
            self.store.update_experiment(
                str(experiment["id"]), status="completed", responses={"adhesion": 8.5}
            )
        with self.assertRaisesRegex(ValueError, "zero is not a valid"):
            self.store.update_experiment(
                str(experiment["id"]),
                status="completed",
                responses={"adhesion": 0, "viscosity_cp": 2350, "dry_time_min": 35, "gloss": 89},
            )

    def test_target_replicates_and_batch_close(self):
        batch = self.store.create_batch(
            self.project, one_recommendation(self.config), decision="RUN", decision_reason="test"
        )
        self.store.approve_batch(batch)
        experiment = self.store.list_experiments(self.project, batch_id=batch).iloc[0]
        created = self.store.ensure_replicate_count(str(experiment["id"]), 3)
        self.assertEqual(len(created), 2)
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.store.ensure_replicate_count(str(experiment["id"]), 3)
        result = self.store.close_batch(batch, cancel_unresolved=True)
        self.assertEqual(result["cancelled"], 3)
        self.assertEqual(self.store.get_batch(batch)["status"], "closed")
        experiments = self.store.list_experiments(self.project, batch_id=batch)
        self.assertTrue((experiments["status"] == "cancelled").all())

    def test_run_and_formulation_calibration_are_separate(self):
        batch = self.store.create_batch(
            self.project, one_recommendation(self.config), decision="RUN", decision_reason="test"
        )
        self.store.approve_batch(batch)
        experiment = self.store.list_experiments(self.project, batch_id=batch).iloc[0]
        self.store.ensure_replicate_count(str(experiment["id"]), 3)
        group = self.store.replicate_group_status(str(experiment["id"]))
        for index, (_, row) in enumerate(group.iterrows()):
            self.store.update_experiment(
                str(row["id"]),
                status="completed",
                qualification_stage="confirmation",
                responses={
                    "adhesion": 8.4 + 0.1 * index,
                    "viscosity_cp": 2340 + 10 * index,
                    "dry_time_min": 35,
                    "gloss": 89,
                },
            )
        report = calibration_report(self.store, self.project)
        run_adhesion = report["run_level"]["response_summary"]
        run_n = int(run_adhesion[run_adhesion["response"] == "adhesion"].iloc[0]["n"])
        form_adhesion = report["formulation_level"]["response_summary"]
        form_n = int(form_adhesion[form_adhesion["response"] == "adhesion"].iloc[0]["n"])
        self.assertEqual(run_n, 3)
        self.assertEqual(form_n, 1)

    def test_robustness_reports_comparable_nominal_probability(self):
        candidate = dict(self.config["baseline"])
        candidate["probability_all_specs"] = 0.82
        result = simulate_manufacturing_variation(
            self.store.project_dataframe(self.project),
            config=self.config,
            candidate=candidate,
            variation_std={"mix_temperature_c": 1.0, "mix_time_min": 1.0},
            n_simulations=100,
            random_state=8,
        )
        self.assertIn("monte_carlo_nominal_success_probability", result)
        self.assertIn("optimizer_nominal_success_probability", result)
        self.assertGreaterEqual(result["monte_carlo_nominal_success_probability"], 0)
        self.assertLessEqual(result["monte_carlo_nominal_success_probability"], 1)


class V062SignatureTests(unittest.TestCase):
    def test_duplicate_unscoped_signature_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PilotStore(Path(tmp) / "pilot.db")
            owner, org = store.register_owner(
                email="owner@example.com",
                display_name="Owner",
                password="SecurePass123",
                organization_name="Lab",
            )
            data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
            config = config_for(data)
            project = store.create_project(
                "Project", config, organization_id=org, created_by_user_id=owner
            )
            evidence = compute_evidence_hash(store, project)
            kwargs = dict(
                project_id=project,
                stage="discovery",
                signer_user_id=owner,
                typed_name="Owner",
                password="SecurePass123",
                signature_meaning="Reviewed.",
                evidence_hash=evidence,
            )
            store.sign_approval(**kwargs)
            with self.assertRaisesRegex(ValueError, "already signed"):
                store.sign_approval(**kwargs)


if __name__ == "__main__":
    unittest.main()
