"""Regression tests for public-demo seeding (demo_seed.py) and the demo catalog."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_catalog import DEMO_DATASETS, SANDBOX_KEYS, create_demo_project, demo_dataset  # noqa: E402
from demo_seed import DEMO_ORGANIZATION, DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD, seed_demo  # noqa: E402
from pilot_store import PilotStore  # noqa: E402


class DemoSeedTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tempdir.name) / "demo.db")

    def tearDown(self):
        self.tempdir.cleanup()

    def _demo_projects(self) -> pd.DataFrame:
        owner = self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)
        organization_id = str(self.store.user_organizations(owner["id"]).iloc[0]["id"])
        return self.store.list_projects(organization_id)

    def test_seed_creates_workspace_with_both_sandbox_projects(self):
        project_id = seed_demo(self.store)
        self.assertIsNotNone(project_id)
        self.assertTrue(self.store.has_users())
        user = self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)
        self.assertIsNotNone(user)
        projects = self._demo_projects()
        self.assertEqual(
            sorted(projects["name"].astype(str)),
            sorted(demo_dataset(key).project_name for key in SANDBOX_KEYS),
        )
        # The first project created is the cosmetics swap, as before.
        data = self.store.project_dataframe(project_id)
        self.assertEqual(len(data), 88)
        config = self.store.get_project(project_id)["config"]
        self.assertEqual(config["ingredient_to_remove"], "ingredient_legacy_peg_emulsifier")
        self.assertEqual(len(config["response_specs"]), 4)
        # And the food project is there with its own history and specifications.
        food = projects[projects["name"] == demo_dataset("food").project_name].iloc[0]
        food_config = self.store.get_project(str(food["id"]))["config"]
        self.assertEqual(food_config["ingredient_to_remove"], "ingredient_legacy_red3")
        self.assertEqual(int(food["experiment_count"]), 90)

    def test_seed_is_idempotent_and_never_touches_populated_databases(self):
        first = seed_demo(self.store)
        self.assertIsNotNone(first)
        self.assertIsNone(seed_demo(self.store))
        organizations = self.store.user_organizations(
            self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)["id"]
        )
        self.assertEqual(len(organizations), 1)
        self.assertEqual(len(self._demo_projects()), len(SANDBOX_KEYS))

    def test_seed_leaves_a_real_workspace_alone(self):
        self.store.register_owner(
            email="owner@example.com",
            display_name="Real Owner",
            password="SecurePass123",
            organization_name="Real Lab",
        )
        self.assertIsNone(seed_demo(self.store))
        self.assertIsNone(self.store.find_user_by_email(DEMO_OWNER_EMAIL))

    def test_seed_completes_an_older_sandbox_without_a_wipe(self):
        # A deployment seeded before the food dataset existed has the owner, the
        # organization and the cosmetics project. A redeploy adds only what is missing.
        user_id, organization_id = self.store.register_owner(
            email=DEMO_OWNER_EMAIL,
            display_name="Demo Explorer",
            password=DEMO_OWNER_PASSWORD,
            organization_name=DEMO_ORGANIZATION,
        )
        cosmetics = create_demo_project(
            self.store, "cosmetics", organization_id=organization_id, created_by_user_id=user_id
        )
        added = seed_demo(self.store)
        self.assertIsNotNone(added)
        self.assertNotEqual(added, cosmetics)
        projects = self._demo_projects()
        self.assertEqual(len(projects), len(SANDBOX_KEYS))
        self.assertEqual(int(projects[projects["id"] == cosmetics].iloc[0]["experiment_count"]), 88)
        self.assertEqual(self.store.get_project(added)["config"]["ingredient_to_remove"], "ingredient_legacy_red3")
        # Running it again adds nothing.
        self.assertIsNone(seed_demo(self.store))
        self.assertEqual(len(self._demo_projects()), len(SANDBOX_KEYS))

    def test_find_user_by_email_does_not_count_as_a_sign_in_attempt(self):
        seed_demo(self.store)
        for _ in range(10):
            self.assertIsNotNone(self.store.find_user_by_email(DEMO_OWNER_EMAIL))
        self.assertIsNone(self.store.find_user_by_email("nobody@example.com"))
        self.assertIsNone(self.store.find_user_by_email("not an email"))
        self.assertIsNotNone(self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD))


class DemoCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tempdir.name) / "catalog.db")
        self.user_id, self.organization_id = self.store.register_owner(
            email="owner@example.com",
            display_name="Owner",
            password="SecurePass123",
            organization_name="Lab",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_every_dataset_ships_and_matches_its_configuration(self):
        for dataset in DEMO_DATASETS:
            with self.subTest(dataset=dataset.key):
                self.assertTrue(dataset.path.exists(), dataset.filename)
                data = dataset.load()
                config = dataset.build_config(data)
                mixture = config["mixture_columns"]
                self.assertEqual(mixture, [c for c in data.columns if c.startswith("ingredient_")])
                self.assertIn(config["ingredient_to_remove"], mixture)
                self.assertEqual(config["mixture_bounds"][config["ingredient_to_remove"]], [0.0, 0.0])
                sums = data[mixture].sum(axis=1)
                self.assertTrue(((sums - config["mixture_total"]).abs() < 0.01).all())
                for spec in config["response_specs"]:
                    self.assertIn(spec["response"], data.columns)
                for column in config["process_columns"] + config["categorical_columns"]:
                    self.assertIn(column, data.columns)
                for column, (low, high) in config["mixture_bounds"].items():
                    self.assertIn(column, mixture)
                    self.assertLessEqual(low, high, column)
                self.assertIn("manufacturing_variation", config)

    def test_food_history_lies_inside_its_design_bounds(self):
        dataset = demo_dataset("food")
        data = dataset.load()
        config = dataset.build_config(data)
        for column, (low, high) in config["mixture_bounds"].items():
            if column == config["ingredient_to_remove"]:
                self.assertTrue((data[column][data["ingredient_legacy_red3"] == 0] == 0).all())
                continue
            self.assertLessEqual(low, data[column].min() + 1e-9, column)
            self.assertGreaterEqual(high, data[column].max() - 1e-9, column)
        for column, (low, high) in config["process_bounds"].items():
            self.assertLessEqual(low, data[column].min() + 1e-9, column)
            self.assertGreaterEqual(high, data[column].max() - 1e-9, column)

    def test_food_dataset_tells_the_red3_story(self):
        data = demo_dataset("food").load()
        self.assertEqual(len(data), 90)
        completed = data[data["status"] == "completed"]
        legacy = completed[completed["ingredient_legacy_red3"] > 0]
        candidates = completed[completed["ingredient_legacy_red3"] == 0]
        failed = data[data["status"] == "failed"]
        self.assertEqual(len(legacy), 18)
        self.assertEqual(len(failed), 8)
        self.assertTrue(failed["failure_reason"].notna().all())
        self.assertTrue(failed["delta_e_vs_standard"].isna().all())
        # Legacy lots sit on the color standard; most trials do not, yet the answer exists.
        self.assertLess(legacy["delta_e_vs_standard"].max(), 3.0)
        self.assertGreater(legacy["a_star"].min(), 38.0)
        meets_all = candidates[
            (candidates["delta_e_vs_standard"] <= 3.0)
            & (candidates["a_star"] >= 38.0)
            & (candidates["color_retention_pct"] >= 85.0)
            & (candidates["flavor_score"] >= 7.5)
        ]
        self.assertGreaterEqual(len(meets_all), 1)
        self.assertLess(len(meets_all), len(candidates) // 4)
        self.assertTrue((meets_all["ingredient_substitute_black_carrot"] > meets_all["ingredient_substitute_beet"]).all())

    def test_food_project_imports_with_lot_codes_and_recommends_a_batch(self):
        from closed_loop import create_recommendation_batch

        project_id = create_demo_project(
            self.store, "food", organization_id=self.organization_id, created_by_user_id=self.user_id
        )
        experiments = self.store.list_experiments(project_id)
        self.assertEqual(len(experiments), 90)
        self.assertIn("LOT-0001", set(experiments["experiment_code"].astype(str)))
        self.assertEqual(int((experiments["status"] == "failed").sum()), 8)
        _, batch_id = create_recommendation_batch(self.store, project_id)
        proposed = self.store.list_experiments(project_id, batch_id=batch_id)
        self.assertEqual(len(proposed), 5)
        # Red No. 3 is gone from every proposal and the mixture still closes to 100.
        self.assertTrue((proposed["ingredient_legacy_red3"].astype(float) == 0.0).all())
        mixture = [c for c in proposed.columns if c.startswith("ingredient_")]
        self.assertTrue(((proposed[mixture].astype(float).sum(axis=1) - 100.0).abs() < 0.05).all())


if __name__ == "__main__":
    unittest.main()
