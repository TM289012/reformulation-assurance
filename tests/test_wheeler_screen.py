"""Tests for the v0.10.0 replicate consistency screen (after Donald Wheeler).

The screen judges each replicate against natural limits (mean ± 2.66 × average
moving range) computed from the other replicates in run order. Only groups
whose replicates agree are scored on CV by the qualification gates.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assurance_v4 import replicate_summary, wheeler_screen  # noqa: E402
from closed_loop import qualification_progress  # noqa: E402
from demo_seed import seed_demo  # noqa: E402
from pilot_store import PilotStore  # noqa: E402


class WheelerScreenUnitTests(unittest.TestCase):
    def test_consistent_replicates_pass(self):
        consistent, flagged = wheeler_screen([10.1, 10.3, 9.9, 10.2, 10.0])
        self.assertTrue(consistent)
        self.assertIsNone(flagged)

    def test_planted_outlier_is_flagged(self):
        # Four tight values and one far outside the noise of the others.
        consistent, flagged = wheeler_screen([10.1, 10.2, 10.0, 10.1, 14.0])
        self.assertFalse(consistent)
        self.assertEqual(flagged, 4)

    def test_outlier_position_is_reported(self):
        consistent, flagged = wheeler_screen([25.0, 10.1, 10.2, 10.0, 10.1])
        self.assertFalse(consistent)
        self.assertEqual(flagged, 0)

    def test_too_few_replicates_returns_none(self):
        consistent, flagged = wheeler_screen([10.0, 10.1])
        self.assertIsNone(consistent)
        self.assertIsNone(flagged)

    def test_identical_others_flag_a_different_value(self):
        consistent, flagged = wheeler_screen([10.0, 10.0, 10.0, 12.0])
        self.assertFalse(consistent)
        self.assertEqual(flagged, 3)

    def test_all_identical_values_are_consistent(self):
        consistent, flagged = wheeler_screen([10.0, 10.0, 10.0, 10.0])
        self.assertTrue(consistent)
        self.assertIsNone(flagged)


class WheelerScreenIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tempdir.name) / "workspace.db")
        self.project_id = seed_demo(self.store)
        # Build one confirmation replicate group of 5 with a planted outlier
        # in viscosity_cp on the final replicate.
        project = self.store.get_project(self.project_id)
        config = project["config"]
        baseline = dict(config["baseline"])
        recommendation = pd.DataFrame([baseline])
        batch_id = self.store.create_batch(
            self.project_id,
            recommendation,
            decision="RUN CONFIRMATION",
            decision_reason="test replicate group",
            qualification_stage="confirmation",
        )
        self.store.approve_batch(batch_id)
        experiments = self.store.list_experiments(self.project_id, source_type="recommended")
        experiment_id = str(experiments.iloc[0]["id"])
        self.store.ensure_replicate_count(experiment_id, 5)
        group = self.store.replicate_group_status(experiment_id)
        base_responses = {"ph": 5.3, "stability_score": 9.0, "spreadability": 8.0}
        viscosities = [10100.0, 10250.0, 9980.0, 10120.0, 14950.0]
        for index, (_, run) in enumerate(group.iterrows()):
            self.store.update_experiment(
                str(run["id"]),
                status="completed",
                responses={**base_responses, "viscosity_cp": viscosities[index]},
            )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_replicate_summary_flags_the_planted_outlier(self):
        summary = replicate_summary(self.store, self.project_id)
        self.assertFalse(summary.empty)
        row = summary.iloc[0]
        self.assertIn("consistent_viscosity_cp", summary.columns)
        self.assertFalse(bool(row["consistent_viscosity_cp"]))
        self.assertIn("replicate #5", str(row["screen_note_viscosity_cp"]))
        # The untouched responses stay consistent.
        self.assertTrue(bool(row["consistent_ph"]))

    def test_qualification_progress_still_computes(self):
        progress = qualification_progress(self.store, self.project_id)
        self.assertIn("stage_progress", progress)
        self.assertIn("replicate_summary", progress)
        stage_view = progress["stage_progress"]
        self.assertFalse(stage_view.empty)


if __name__ == "__main__":
    unittest.main()
