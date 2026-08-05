"""Regression tests for public-demo seeding (demo_seed.py)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_seed import DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD, seed_demo  # noqa: E402
from pilot_store import PilotStore  # noqa: E402


class DemoSeedTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tempdir.name) / "demo.db")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_seed_creates_workspace_with_preloaded_project(self):
        project_id = seed_demo(self.store)
        self.assertIsNotNone(project_id)
        self.assertTrue(self.store.has_users())
        user = self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)
        self.assertIsNotNone(user)
        data = self.store.project_dataframe(project_id)
        self.assertEqual(len(data), 88)
        config = self.store.get_project(project_id)["config"]
        self.assertEqual(config["ingredient_to_remove"], "ingredient_legacy_peg_emulsifier")
        self.assertEqual(len(config["response_specs"]), 4)

    def test_seed_is_idempotent_and_never_touches_populated_databases(self):
        first = seed_demo(self.store)
        self.assertIsNotNone(first)
        self.assertIsNone(seed_demo(self.store))
        organizations = self.store.user_organizations(
            self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)["id"]
        )
        self.assertEqual(len(organizations), 1)


if __name__ == "__main__":
    unittest.main()
