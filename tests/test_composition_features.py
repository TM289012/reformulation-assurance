"""Boundary-aware feature representations (v0.12.9).

Pins the properties the matsci.org reviewers asked for: presence indicators only
where an ingredient is sometimes absent, an ilr basis that is a true isometry,
zero replacement that keeps every row's total, and a default path that is
unchanged byte for byte so the app's behaviour does not move.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from composition_features import (  # noqa: E402
    CompositionFeatures,
    CompositionSpec,
    composition_spec_from_config,
    ilr_balances,
    multiplicative_replacement,
    sequential_binary_partition,
)
from demo_catalog import cosmetics_config  # noqa: E402
from reformulation_engine import MixedFeatureEncoder, _fit_response_model  # noqa: E402

COLUMNS = ("legacy", "sub_a", "sub_b", "water", "oil")


def _clr(values: np.ndarray) -> np.ndarray:
    logs = np.log(values)
    return logs - logs.mean(axis=1, keepdims=True)


class PartitionAndIlr(unittest.TestCase):
    def test_partition_puts_the_slot_first_and_covers_every_column_once(self):
        balances = sequential_binary_partition(COLUMNS, slot=("legacy", "sub_a", "sub_b"), removed="legacy")
        self.assertEqual(len(balances), len(COLUMNS) - 1)
        self.assertEqual(balances[0], (("legacy", "sub_a", "sub_b"), ("water", "oil")))
        self.assertEqual(balances[1], (("legacy",), ("sub_a", "sub_b")))
        self.assertEqual(balances[2], (("sub_a",), ("sub_b",)))
        self.assertEqual(balances[3], (("water",), ("oil",)))
        for numerator, denominator in balances:
            self.assertFalse(set(numerator) & set(denominator))

    def test_partition_without_a_slot_is_sequential(self):
        balances = sequential_binary_partition(("a", "b", "c"), slot=(), removed=None)
        self.assertEqual(balances, [(("a",), ("b", "c")), (("b",), ("c",))])

    def test_ilr_is_an_isometry_of_the_aitchison_geometry(self):
        rng = np.random.default_rng(3)
        X = rng.dirichlet(np.ones(len(COLUMNS)), size=12) * 100.0
        balances = sequential_binary_partition(COLUMNS, slot=("legacy", "sub_a", "sub_b"), removed="legacy")
        Z = ilr_balances(X, COLUMNS, balances)
        C = _clr(X)
        for i in range(len(X)):
            for j in range(len(X)):
                self.assertAlmostEqual(
                    np.linalg.norm(Z[i] - Z[j]), np.linalg.norm(C[i] - C[j]), places=9
                )

    def test_ilr_of_a_scaled_row_is_unchanged(self):
        balances = sequential_binary_partition(COLUMNS, slot=("legacy",), removed="legacy")
        row = np.array([[2.0, 3.0, 5.0, 70.0, 20.0]])
        np.testing.assert_allclose(ilr_balances(row, COLUMNS, balances), ilr_balances(row * 3, COLUMNS, balances))

    def test_ilr_refuses_zeros(self):
        balances = sequential_binary_partition(COLUMNS, slot=("legacy",), removed="legacy")
        with self.assertRaises(ValueError):
            ilr_balances(np.array([[0.0, 3.0, 5.0, 72.0, 20.0]]), COLUMNS, balances)


class ZeroReplacement(unittest.TestCase):
    def test_zeros_become_deltas_and_totals_are_kept(self):
        X = np.array([[0.0, 3.0, 0.0, 77.0, 20.0], [2.0, 3.0, 5.0, 70.0, 20.0]])
        deltas = np.array([0.5, 0.1, 0.2, 1.0, 1.0])
        R = multiplicative_replacement(X, deltas, total=100.0)
        np.testing.assert_allclose(R.sum(axis=1), [100.0, 100.0])
        self.assertEqual(R[0, 0], 0.5)
        self.assertEqual(R[0, 2], 0.2)
        self.assertTrue(np.all(R > 0))
        np.testing.assert_allclose(R[1], X[1])  # a row without zeros is untouched
        # non-zero parts shrink by the same factor
        np.testing.assert_allclose(R[0, [1, 3, 4]] / X[0, [1, 3, 4]], 1.0 - 0.7 / 100.0)


class Features(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "legacy": [2.0, 0.0, 1.5, 0.0],
                "sub_a": [1.0, 2.0, 1.5, 2.5],
                "sub_b": [0.0, 1.0, 0.0, 0.5],
                "water": [77.0, 77.0, 77.0, 77.0],
                "oil": [20.0, 20.0, 20.0, 20.0],
            }
        )
        self.frame["water"] = 100.0 - self.frame[["legacy", "sub_a", "sub_b", "oil"]].sum(axis=1)

    def test_indicators_only_where_an_ingredient_is_sometimes_absent(self):
        spec = CompositionSpec(COLUMNS, removed="legacy", optional=("legacy", "sub_a", "sub_b"), representation="presence")
        features = CompositionFeatures(spec).fit(self.frame)
        self.assertEqual(features.indicator_columns, ["legacy", "sub_b"])  # sub_a is always present
        out = features.transform(self.frame)
        self.assertEqual(list(out.columns), ["present_legacy", "present_sub_b"])
        self.assertEqual(out["present_legacy"].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(out["present_sub_b"].tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_ilr_representation_adds_d_minus_one_coordinates_and_learns_deltas(self):
        spec = CompositionSpec(COLUMNS, removed="legacy", optional=("legacy", "sub_a", "sub_b"), representation="presence_ilr")
        features = CompositionFeatures(spec).fit(self.frame)
        self.assertEqual(features.feature_names, ["present_legacy", "present_sub_b", "ilr_1", "ilr_2", "ilr_3", "ilr_4"])
        np.testing.assert_allclose(features.deltas[0], 0.65 * 1.5)  # 65% of the smallest non-zero legacy amount
        np.testing.assert_allclose(features.deltas[2], 0.65 * 0.5)
        out = features.transform(self.frame)
        self.assertTrue(np.all(np.isfinite(out.to_numpy())))
        new = pd.DataFrame({"legacy": [0.0], "sub_a": [3.0], "sub_b": [0.0], "water": [77.0], "oil": [20.0]})
        self.assertEqual(features.transform(new).shape, (1, 6))

    def test_raw_representation_is_declared_but_adds_nothing(self):
        spec = CompositionSpec(COLUMNS, representation="raw")
        self.assertFalse(spec.uses_indicators)
        self.assertFalse(spec.uses_ilr)
        encoder = MixedFeatureEncoder(["legacy", "sub_a"], [], composition=spec)
        self.assertIsNone(encoder.composition)

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            CompositionSpec(COLUMNS, representation="clr")
        with self.assertRaises(ValueError):
            CompositionSpec(COLUMNS, removed="not_a_column")
        with self.assertRaises(ValueError):
            CompositionSpec(COLUMNS, optional=("nope",))

    def test_spec_from_config_reads_the_slot_off_the_bounds(self):
        data = pd.read_csv(ROOT / "demo_cosmetics_emulsifier_swap.csv")
        spec = composition_spec_from_config(cosmetics_config(data), "presence")
        self.assertEqual(spec.removed, "ingredient_legacy_peg_emulsifier")
        self.assertEqual(
            spec.optional,
            (
                "ingredient_legacy_peg_emulsifier",
                "ingredient_substitute_polyglyceryl",
                "ingredient_substitute_glucoside",
            ),
        )
        self.assertEqual(spec.total, 100.0)


class EncoderAndModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "demo_cosmetics_emulsifier_swap.csv")
        cls.config = cosmetics_config(cls.data)
        cls.numeric = [*cls.config["mixture_columns"], *cls.config["process_columns"]]
        cls.categorical = list(cls.config["categorical_columns"])
        cls.features = [*cls.numeric, *cls.categorical]

    def test_default_encoder_is_unchanged(self):
        plain = MixedFeatureEncoder(self.numeric, self.categorical).fit(self.data[self.features])
        raw = MixedFeatureEncoder(
            self.numeric, self.categorical, composition=composition_spec_from_config(self.config, "raw")
        ).fit(self.data[self.features])
        np.testing.assert_array_equal(plain.transform(self.data[self.features]), raw.transform(self.data[self.features]))
        self.assertEqual(plain.numeric_feature_names, self.numeric)

    def test_presence_widens_by_the_informative_indicators(self):
        plain = MixedFeatureEncoder(self.numeric, self.categorical).fit(self.data[self.features])
        presence = MixedFeatureEncoder(
            self.numeric, self.categorical, composition=composition_spec_from_config(self.config, "presence")
        ).fit(self.data[self.features])
        width = plain.transform(self.data[self.features]).shape[1]
        self.assertEqual(presence.transform(self.data[self.features]).shape[1], width + 2)
        self.assertEqual(
            presence.composition_features.indicator_columns,
            ["ingredient_legacy_peg_emulsifier", "ingredient_substitute_glucoside"],
        )

    def test_ilr_replaces_the_mixture_columns(self):
        plain = MixedFeatureEncoder(self.numeric, self.categorical).fit(self.data[self.features])
        ilr = MixedFeatureEncoder(
            self.numeric, self.categorical, composition=composition_spec_from_config(self.config, "presence_ilr")
        ).fit(self.data[self.features])
        width = plain.transform(self.data[self.features]).shape[1]
        n_mix = len(self.config["mixture_columns"])
        self.assertEqual(ilr.transform(self.data[self.features]).shape[1], width - n_mix + 2 + (n_mix - 1))
        self.assertEqual(ilr.raw_numeric_columns, list(self.config["process_columns"]))

    def test_fit_response_model_default_path_has_no_feature_note(self):
        _, notes = _fit_response_model(self.data, self.features, self.numeric, self.categorical, "ph", 42)
        self.assertFalse(any(note.startswith("Features for") for note in notes))

    def test_fit_response_model_with_a_representation(self):
        spec = composition_spec_from_config(self.config, "presence_ilr")
        model, notes = _fit_response_model(
            self.data, self.features, self.numeric, self.categorical, "ph", 42, composition=spec
        )
        self.assertTrue(any("representation=presence_ilr" in note for note in notes))
        mean, std, _ = model.predict(self.data[self.features].head(5))
        self.assertEqual(mean.shape, (5,))
        self.assertTrue(np.all(std > 0))

    def test_fit_response_model_insists_on_the_mixture_columns(self):
        spec = composition_spec_from_config(self.config, "presence")
        partial = [c for c in self.features if c != "ingredient_legacy_peg_emulsifier"]
        with self.assertRaises(ValueError):
            _fit_response_model(self.data, partial, partial[:-1], self.categorical, "ph", 42, composition=spec)


class BenchmarkArms(unittest.TestCase):
    def test_arm_registry_and_history_subsample(self):
        import backtest_ranker_vs_baybe as bench

        self.assertEqual(bench.OURS_ARMS["ours"], "raw")
        self.assertEqual(set(bench.OURS_ARMS), {"ours", "ours_presence", "ours_presence_ilr"})
        data, _ = bench.load_cosmetics()
        small = bench.subsample_history(data, 30, seed=1)
        self.assertEqual(len(small), 30)
        self.assertIn("failed", small["status"].str.lower().tolist())
        self.assertTrue(bench.subsample_history(data, None, seed=1) is data)
        self.assertTrue(bench.subsample_history(data, 500, seed=1) is data)
        pd.testing.assert_frame_equal(small, bench.subsample_history(data, 30, seed=1))


if __name__ == "__main__":
    unittest.main()
