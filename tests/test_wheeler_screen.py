"""Tests for the replicate consistency screen (XmR limits from the other replicates) and the chunky-data rule.

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

    def test_identical_siblings_get_limits_widened_for_round_off(self):
        # Two identical readings have a zero moving range, so limits built from them
        # alone would have no width. The group as a whole (moving ranges 0 and 0.3,
        # five possible values within the limits) is not chunky; the third value is
        # judged against limits widened by the most round-off could hide
        # (2.66 x (0 + 0.1) + 0.1 = 0.366), and 0.3 away is inside them.
        from assurance_v4 import wheeler_screen_detail

        detail = wheeler_screen_detail([8.2, 8.2, 8.5])
        self.assertTrue(detail["consistent"])
        self.assertIsNone(detail["flagged"])
        self.assertEqual(detail["widened"], [2])
        self.assertFalse(detail["chunky"])
        self.assertEqual(detail["possible_values"], 5)

    def test_a_wild_value_beside_identical_siblings_is_caught(self):
        # 0.5 away from two identical readings is outside even the widened limits.
        from assurance_v4 import wheeler_screen_detail

        detail = wheeler_screen_detail([5.4, 5.4, 5.9])
        self.assertFalse(detail["consistent"])
        self.assertEqual(detail["flagged"], 2)
        self.assertIn(2, detail["widened"])
        # Five identical readings and one ten steps away: caught the same way.
        detail = wheeler_screen_detail([5.4, 5.4, 5.4, 5.4, 5.4, 6.4])
        self.assertFalse(detail["consistent"])
        self.assertEqual(detail["flagged"], 5)

    def test_one_step_difference_after_identical_readings_is_not_a_flag(self):
        # A pH meter reading 5.4, 5.4 then 5.3: v0.12.0 flagged the 5.3. It is not an
        # inconsistent replicate; it is chunky data (Wheeler), so nothing is judged.
        consistent, flagged = wheeler_screen([5.4, 5.4, 5.3])
        self.assertIsNot(consistent, False)
        self.assertIsNone(flagged)

    def test_spread_among_siblings_still_catches_a_wild_value(self):
        consistent, flagged = wheeler_screen([5.4, 5.3, 5.4, 6.9])
        self.assertFalse(consistent)
        self.assertEqual(flagged, 3)

    def test_all_identical_values_are_chunky(self):
        # Nothing is inconsistent, but the readings cannot show their own spread.
        from assurance_v4 import wheeler_screen_detail

        consistent, flagged = wheeler_screen([10.0, 10.0, 10.0, 10.0])
        self.assertTrue(consistent)
        self.assertIsNone(flagged)
        self.assertTrue(wheeler_screen_detail([10.0, 10.0, 10.0, 10.0])["chunky"])


class ChunkyDataTests(unittest.TestCase):
    """Wheeler, "What is Chunky Data?" (2011): for a moving-range chart, three or fewer
    possible range values inside the range limits mean the recording increment hides the
    routine variation; four is the borderline-safe condition."""

    def test_measurement_increment_is_read_off_the_data(self):
        from assurance_v4 import measurement_increment

        self.assertEqual(measurement_increment([5.4, 5.4, 5.3]), 0.1)
        self.assertEqual(measurement_increment([5.42, 5.4]), 0.01)
        self.assertEqual(measurement_increment([5.42, 5.39, 5.36]), 0.01)

    def test_a_step_every_reading_shares_is_the_increment(self):
        from assurance_v4 import measurement_increment

        # Viscosity logged in steps of 50: the decimal place says 1, the data say 50.
        logged = [11250.0, 11300.0, 11250.0, 11200.0, 11250.0, 11300.0, 11350.0, 11250.0]
        self.assertEqual(measurement_increment(logged), 50.0)
        self.assertEqual(measurement_increment([11250.0, 11400.0]), 50.0)
        self.assertEqual(measurement_increment([5.35, 5.40, 5.45, 5.35, 5.40, 5.50, 5.45, 5.40]), 0.05)
        self.assertEqual(measurement_increment([35.0, 35.0, 40.0]), 5.0)
        # One reading off the step and the step is gone.
        self.assertEqual(measurement_increment(logged[:-1] + [11251.0]), 1.0)
        # Readings that share a step by chance are read coarser than they were (the
        # fail-safe direction); a declared recording step corrects it.
        self.assertEqual(measurement_increment([10.0, 10.0, 5.0]), 5.0)

    def test_logged_in_steps_of_fifty_the_series_is_chunky(self):
        from assurance_v4 import chunky_data_check, measurement_increment

        logged = [11250.0, 11300.0, 11250.0, 11250.0, 11300.0, 11250.0, 11250.0, 11300.0, 11250.0, 11250.0]
        check = chunky_data_check(logged, measurement_increment(logged))
        self.assertEqual(check["increment"], 50.0)
        self.assertTrue(check["chunky"])

    def test_temperatures_to_the_nearest_degree_are_chunky(self):
        from assurance_v4 import chunky_data_check

        # Average moving range 0.8 degree: upper range limit 2.6, possible values 0, 1, 2 -> three.
        check = chunky_data_check([18, 19, 18, 18, 19, 18])
        self.assertEqual(check["possible_values"], 3)
        self.assertTrue(check["chunky"])
        # The same readings with one more digit are fine.
        check = chunky_data_check([18.2, 19.1, 18.4, 18.3, 18.9, 18.1])
        self.assertFalse(check["chunky"])
        self.assertGreaterEqual(check["possible_values"], 5)

    def test_identical_siblings_are_the_extreme_case(self):
        from assurance_v4 import chunky_data_check

        check = chunky_data_check([10.0, 10.0])
        self.assertEqual(check["possible_values"], 1)
        self.assertTrue(check["chunky"])

    def test_the_case_wheeler_answered_is_chunky_and_not_flagged(self):
        from assurance_v4 import wheeler_screen_detail

        # Moving ranges 0 and 0.1: upper range limit 0.16, possible values 0 and 0.1.
        # v0.12.0 flagged the 5.3; against limits widened for round-off it is not flagged.
        detail = wheeler_screen_detail([5.4, 5.4, 5.3])
        self.assertTrue(detail["consistent"])
        self.assertTrue(detail["chunky"])
        self.assertEqual(detail["possible_values"], 2)
        self.assertEqual(detail["widened"], [2])
        self.assertEqual(detail["increment"], 0.1)
        # One more recorded digit and the same readings are judged normally.
        detail = wheeler_screen_detail([5.42, 5.39, 5.36])
        self.assertTrue(detail["consistent"])
        self.assertFalse(detail["chunky"])
        self.assertEqual(detail["widened"], [])

    def test_a_wild_value_inside_a_chunky_group_is_caught(self):
        from assurance_v4 import wheeler_screen_detail

        # Five identical readings and one four steps away: three possible values, chunky,
        # and the 5.8 is outside even the widened limits (0.4 > 0.366).
        detail = wheeler_screen_detail([5.4, 5.4, 5.4, 5.4, 5.4, 5.8])
        self.assertTrue(detail["chunky"])
        self.assertFalse(detail["consistent"])
        self.assertEqual(detail["flagged"], 5)

    def test_a_declared_step_overrides_the_reading(self):
        from assurance_v4 import cv_upper_bound, wheeler_screen_detail

        # pH 7.00 three times is stored as 7.0: read off the numbers the step looks like 1.
        self.assertEqual(wheeler_screen_detail([7.0, 7.0, 7.0])["increment"], 1.0)
        detail = wheeler_screen_detail([7.0, 7.0, 7.0], increment=0.01)
        self.assertEqual(detail["increment"], 0.01)
        self.assertTrue(detail["chunky"])
        self.assertLess(cv_upper_bound([7.0, 7.0, 7.0], 0.01), 0.001)

    def test_the_rule_is_applied_to_the_whole_group(self):
        from assurance_v4 import wheeler_screen_detail

        # Three identical readings and one two steps away: moving ranges 0, 0, 2, so
        # three possible values within the limits. Chunky, so nothing is judged.
        detail = wheeler_screen_detail([10.0, 10.0, 10.0, 12.0])
        self.assertTrue(detail["chunky"])
        self.assertEqual(detail["possible_values"], 3)
        self.assertIsNone(detail["flagged"])

    def test_identical_readings_in_a_pair_are_chunky(self):
        from assurance_v4 import wheeler_screen_detail

        detail = wheeler_screen_detail([5.4, 5.4])
        self.assertTrue(detail["chunky"])
        self.assertIsNone(detail["consistent"])
        detail = wheeler_screen_detail([5.4, 5.3])
        self.assertFalse(detail["chunky"])

    def test_a_wild_value_is_still_caught_when_its_siblings_are_fine(self):
        from assurance_v4 import wheeler_screen_detail

        detail = wheeler_screen_detail([11250.0, 11400.0, 12800.0])
        self.assertFalse(detail["consistent"])
        self.assertEqual(detail["flagged"], 2)
        self.assertFalse(detail["chunky"])


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
        # The untouched responses are not called inconsistent. Recorded identically five
        # times they are chunky data instead (see test_summary_names_chunky_responses).
        self.assertIsNot(row["consistent_ph"], False)
        self.assertNotIn("inconsistent", str(row["screen_note_ph"]))

    def test_qualification_progress_still_computes(self):
        progress = qualification_progress(self.store, self.project_id)
        self.assertIn("stage_progress", progress)
        self.assertIn("replicate_summary", progress)
        stage_view = progress["stage_progress"]
        self.assertFalse(stage_view.empty)

    def test_summary_names_chunky_responses(self):
        # pH was recorded as 5.3 five times: identical readings show the increment, not agreement.
        summary = replicate_summary(self.store, self.project_id)
        row = summary.iloc[0]
        self.assertTrue(bool(row["chunky_ph"]))
        self.assertIn("record one more digit", str(row["screen_note_ph"]))
        self.assertIn("more replicates will not reliably fix it", str(row["screen_note_ph"]))
        self.assertFalse(bool(row["chunky_viscosity_cp"]))


class PaperNumbersTests(unittest.TestCase):
    """The counting convention checked against Wheeler's own worked numbers."""

    def test_counts_match_the_paper(self):
        from assurance_v4 import possible_range_values

        # Figure 3: upper range limit .01810 at an increment of .001, "19 possible values".
        self.assertEqual(possible_range_values(0.01810, 0.001), 19)
        # Figure 4: upper range limit .0102 at .01, "only two possible values".
        self.assertEqual(possible_range_values(0.0102, 0.01), 2)
        # Table 3, n = 2, SD(X) equal to one increment: limit 3.69, values 0, 1, 2, 3 (four),
        # the borderline-safe condition.
        self.assertEqual(possible_range_values(3.268 * 1.128, 1.0), 4)


class CVUpperBoundTests(unittest.TestCase):
    def test_known_value(self):
        from assurance_v4 import cv_upper_bound

        # s = 0.0577, plus 0.05 x sqrt(3/2) = 0.0612; mean 5.367 less one step = 5.267.
        self.assertAlmostEqual(cv_upper_bound([5.4, 5.4, 5.3], 0.1), 0.02259, places=4)
        # A mean within one step of zero cannot be bounded.
        self.assertEqual(cv_upper_bound([0.05, 0.05, 0.1], 0.1), float("inf"))

    def test_bound_holds_for_rounded_and_truncated_readings(self):
        import math
        import random

        from assurance_v4 import cv_upper_bound

        def cv(values):
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            return sd / abs(mean)

        rng = random.Random(20260928)
        for _ in range(4000):
            n = rng.randint(2, 8)
            step = rng.choice([0.1, 0.01, 1.0, 50.0])
            center = rng.uniform(20, 200) * step
            spread = rng.uniform(0.05, 3.0) * step
            true_values = [rng.gauss(center, spread) for _ in range(n)]
            for record in (lambda v: round(v / step) * step, lambda v: math.floor(v / step) * step):
                readings = [record(v) for v in true_values]
                self.assertGreaterEqual(cv_upper_bound(readings, step) + 1e-12, cv(true_values))


class ChunkyGateTests(unittest.TestCase):
    """Chunky replicate groups are not screened and are judged on the largest CV their
    readings allow: they pass only when even that bound meets the limit."""

    def _group(self, ph_values, stability=(9.2, 9.1, 9.3), recording_steps=None):
        tempdir = tempfile.TemporaryDirectory()
        store = PilotStore(Path(tempdir.name) / "workspace.db")
        project_id = seed_demo(store)
        config = store.get_project(project_id)["config"]
        if recording_steps:
            store.update_project_config(project_id, {**config, "recording_steps": recording_steps})
            config = store.get_project(project_id)["config"]
        batch_id = store.create_batch(
            project_id, pd.DataFrame([dict(config["baseline"])]), decision="RUN CONFIRMATION",
            decision_reason="chunky test", qualification_stage="confirmation",
        )
        store.approve_batch(batch_id)
        experiment_id = str(store.list_experiments(project_id, source_type="recommended").iloc[0]["id"])
        store.ensure_replicate_count(experiment_id, 3)
        group = store.replicate_group_status(experiment_id)
        viscosities = [10850.0, 10720.0, 10930.0]
        spread = [8.3, 8.5, 8.4]
        for index, (_, run) in enumerate(group.iterrows()):
            store.update_experiment(
                str(run["id"]), status="completed", qualification_stage="confirmation",
                responses={"viscosity_cp": viscosities[index], "ph": ph_values[index],
                           "stability_score": stability[index], "spreadability": spread[index]},
            )
        progress = qualification_progress(store, project_id)
        stage = progress["stage_progress"]
        row = stage[stage["stage"] == "confirmation"].iloc[0]
        tempdir.cleanup()
        return row, progress["replicate_summary"].iloc[0]

    def test_coarse_ph_is_chunky_but_its_bound_meets_the_limit(self):
        # pH 5.4, 5.4, 5.3: chunky, not screened. The largest CV these readings allow is
        # about 2.3%, inside the 8% limit, so the group counts.
        coarse, summary = self._group([5.4, 5.4, 5.3])
        self.assertTrue(bool(summary["chunky_ph"]))
        self.assertTrue(bool(summary["consistent_ph"]))
        self.assertAlmostEqual(float(summary["cv_upper_ph"]), 0.02259, places=4)
        self.assertIn("largest CV these readings allow", str(summary["screen_note_ph"]))
        self.assertEqual(int(coarse["passing_replicate_groups"]), 1)

        fine, fine_summary = self._group([5.42, 5.39, 5.36])
        self.assertFalse(bool(fine_summary["chunky_ph"]))
        self.assertEqual(int(fine["passing_replicate_groups"]), 1)

    def test_whole_number_scores_too_coarse_for_the_limit_block(self):
        # Scores 9, 9, 8 on a whole-number scale: chunky, and the largest CV they allow
        # (about 15.5%) is above the 8% limit, so the group cannot count.
        row, summary = self._group([5.42, 5.39, 5.36], stability=(9.0, 9.0, 8.0))
        self.assertTrue(bool(summary["chunky_stability_score"]))
        self.assertGreater(float(summary["cv_upper_stability_score"]), 0.08)
        self.assertIn("whole-number scores", str(summary["screen_note_stability_score"]))
        self.assertEqual(int(row["passing_replicate_groups"]), 0)
        self.assertIn("record one more digit", str(row["remaining_requirements"]))

    def test_two_identical_readings_do_not_sink_a_group_that_shows_variation(self):
        # 5.40, 5.40, 5.43 at 0.01: moving ranges 0 and 0.03, five possible values within
        # the limits, so not chunky. The third reading is judged against limits widened
        # for round-off, is inside them, and the CV passes.
        row, summary = self._group([5.40, 5.40, 5.43])
        self.assertFalse(bool(summary["chunky_ph"]))
        self.assertIn("widened for round-off", str(summary["screen_note_ph"]))
        self.assertEqual(int(row["passing_replicate_groups"]), 1)

    def test_a_wild_reading_beside_identical_ones_blocks(self):
        # 5.40, 5.40, 5.90: the 5.90 is outside even the widened limits.
        row, summary = self._group([5.40, 5.40, 5.90])
        self.assertIsNotNone(summary["consistent_ph"])
        self.assertFalse(bool(summary["consistent_ph"]))
        self.assertIn("replicate #3 inconsistent", str(summary["screen_note_ph"]))
        self.assertEqual(int(row["passing_replicate_groups"]), 0)

    def test_rounding_cannot_flatter_a_group_that_is_not_chunky(self):
        # Scores 9, 8, 9 on a whole-number scale: four possible values, so not chunky, and a
        # CV of 6.7% looks inside the 8% limit. True values of 9.49, 7.51, 9.49 round to the
        # same readings with a CV of 12.9%, so the gate uses the bound (15.5%) and blocks.
        row, summary = self._group([5.42, 5.39, 5.36], stability=(9.0, 8.0, 9.0))
        self.assertFalse(bool(summary["chunky_stability_score"]))
        self.assertLess(float(summary["cv_stability_score"]), 0.08)
        self.assertGreater(float(summary["cv_upper_stability_score"]), 0.08)
        self.assertEqual(int(row["passing_replicate_groups"]), 0)
        self.assertIn("allow a CV of up to", str(row["remaining_requirements"]))

    def test_declared_steps_decide_both_ways(self):
        # pH 7.00 x3 stored as 7.0: read as a step of 1 the bound is 10.2%, over the limit;
        # declared as 0.01 the bound is under 0.1% and the group counts.
        blocked, _ = self._group([7.0, 7.0, 7.0])
        self.assertEqual(int(blocked["passing_replicate_groups"]), 0)
        passed, summary = self._group([7.0, 7.0, 7.0], recording_steps={"ph": 0.01})
        self.assertIn("set for this project", str(summary["screen_note_ph"]))
        self.assertEqual(int(passed["passing_replicate_groups"]), 1)
        # Scores logged in steps of 5 (35, 35, 35) are read as 5, not 1: the bound (10.2%) blocks.
        coarse, summary = self._group([5.42, 5.39, 5.36], stability=(35.0, 35.0, 35.0))
        self.assertEqual(float(summary["cv_upper_stability_score"]) > 0.08, True)
        self.assertEqual(int(coarse["passing_replicate_groups"]), 0)


if __name__ == "__main__":
    unittest.main()
