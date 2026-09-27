"""v0.12: roster invitations and the workspace overview (course sections, team leads)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from demo_seed import DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD, seed_demo
from pilot_store import PilotStore

ROOT = Path(__file__).resolve().parents[1]


class RosterInvitationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tmp.name) / "course.db")
        self.instructor, self.section = self.store.register_owner(
            email="instructor@university.edu", display_name="Instructor", password="SecurePass123",
            organization_name="CHEM 3410 Fall 2026",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_roster_parsing_accepts_lines_commas_and_angle_brackets(self):
        text = "one@university.edu\nTwo@University.edu, three@university.edu; <four@university.edu>\n\none@university.edu"
        self.assertEqual(
            self.store.parse_email_list(text),
            ["one@university.edu", "two@university.edu", "three@university.edu", "four@university.edu"],
        )
        self.assertEqual(self.store.parse_email_list(""), [])

    def test_roster_invitations_create_one_link_each_and_report_bad_addresses(self):
        roster = self.store.create_invitations(
            self.section,
            emails=["one@university.edu", "two@university.edu", "not-an-email", ""],
            role="scientist", actor_user_id=self.instructor, base_url="https://course.example.test", expires_hours=336,
        )
        self.assertEqual(roster["status"].tolist()[:2], ["invited", "invited"])
        self.assertTrue(roster["status"].iloc[2].startswith("skipped"))
        self.assertEqual(len(roster), 3)
        self.assertTrue(all(url.startswith("https://course.example.test?invite=") for url in roster["invite_url"].iloc[:2]))
        self.assertEqual(len(set(roster["invite_url"].iloc[:2])), 2)

        pending = self.store.list_invitations(self.section, self.instructor)
        self.assertEqual(sorted(pending["email"]), ["one@university.edu", "two@university.edu"])
        self.assertEqual(set(pending["status"]), {"pending"})

        # A student accepts through the link and lands in the section as a scientist.
        token = roster["invite_url"].iloc[0].split("invite=")[1]
        user_id, organization_id = self.store.accept_invitation(token, display_name="Student One", password="SecurePass123")
        self.assertEqual(organization_id, self.section)
        memberships = self.store.user_organizations(user_id)
        self.assertEqual(memberships["role"].tolist(), ["scientist"])

    def test_roster_invitations_require_an_admin(self):
        student = self.store.create_member(
            self.section, email="s@university.edu", display_name="S", password="SecurePass123",
            role="scientist", actor_user_id=self.instructor,
        )
        with self.assertRaises(PermissionError):
            self.store.create_invitations(
                self.section, emails=["x@university.edu"], role="scientist", actor_user_id=student,
            )


class WorkspaceOverviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tmp.name) / "course.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_overview_counts_per_project_and_is_admin_only(self):
        project_id = seed_demo(self.store)
        owner = self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)
        organization_id = str(self.store.user_organizations(owner["id"]).iloc[0]["id"])

        overview = self.store.workspace_overview(organization_id, owner["id"])
        self.assertEqual(len(overview), 1)
        row = overview.iloc[0]
        self.assertEqual(row["id"], project_id)
        self.assertEqual(row["created_by_email"], DEMO_OWNER_EMAIL)
        self.assertEqual(int(row["historical_rows"]), 88)
        self.assertGreaterEqual(int(row["recommended_experiments"]), 0)
        self.assertEqual(int(row["approvals"]), 0)
        self.assertIsInstance(row["last_activity"], str)

        # A second, empty project by another member shows up with zero counts.
        student = self.store.create_member(
            organization_id, email="student@university.edu", display_name="Student", password="SecurePass123",
            role="scientist", actor_user_id=owner["id"],
        )
        data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        from tests.test_v05 import config_for

        second = self.store.create_project(
            "Student project", config_for(data), organization_id=organization_id, created_by_user_id=student,
        )
        overview = self.store.workspace_overview(organization_id, owner["id"])
        self.assertEqual(overview["id"].tolist(), [project_id, second])
        student_row = overview[overview["id"] == second].iloc[0]
        self.assertEqual(student_row["created_by"], "Student")
        self.assertEqual(int(student_row["historical_rows"]), 0)
        self.assertEqual(int(student_row["batches"]), 0)

        with self.assertRaises(PermissionError):
            self.store.workspace_overview(organization_id, student)

    def test_overview_is_empty_for_a_fresh_workspace(self):
        owner, organization_id = self.store.register_owner(
            email="lead@example.com", display_name="Lead", password="SecurePass123", organization_name="Lab",
        )
        overview = self.store.workspace_overview(organization_id, owner)
        self.assertTrue(overview.empty)
        self.assertIn("last_activity", overview.columns)


if __name__ == "__main__":
    unittest.main()
