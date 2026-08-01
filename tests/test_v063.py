"""v0.6.3 security regression tests: organization-scoped outbox.

The v0.6.2 admin outbox rendered the global notification table, which let an
administrator of one organization read another organization's single-use
invitation and password-reset links. These tests pin the fix.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pilot_store import PilotStore


class OutboxScopeFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = PilotStore(self.root / "pilot.db")
        self.owner_a, self.org_a = self.store.register_owner(
            email="owner-a@example.com",
            display_name="Owner A",
            password="SecurePass123",
            organization_name="Org A",
        )
        self.owner_b, self.org_b = self.store.register_owner(
            email="owner-b@example.com",
            display_name="Owner B",
            password="SecurePass123",
            organization_name="Org B",
        )
        self.invite_a = self.store.create_invitation(
            self.org_a,
            email="scientist-a@example.com",
            role="scientist",
            actor_user_id=self.owner_a,
            base_url="https://pilot.example",
        )
        self.invite_b = self.store.create_invitation(
            self.org_b,
            email="scientist-b@example.com",
            role="scientist",
            actor_user_id=self.owner_b,
            base_url="https://pilot.example",
        )
        self.reset = self.store.request_password_reset("owner-b@example.com")

    def tearDown(self):
        self.tmp.cleanup()


class OutboxScopeTests(OutboxScopeFixture):
    def test_outbox_excludes_other_organizations(self):
        outbox_a = self.store.list_outbox(self.org_a, self.owner_a)
        self.assertFalse(outbox_a.empty)
        self.assertTrue((outbox_a["organization_id"] == self.org_a).all())
        recipients = set(outbox_a["recipient_email"])
        self.assertIn("scientist-a@example.com", recipients)
        self.assertNotIn("scientist-b@example.com", recipients)
        bodies = " ".join(outbox_a["body"].astype(str))
        self.assertNotIn(self.invite_b["invite_url"], bodies)

    def test_outbox_never_contains_password_resets(self):
        self.assertIsNotNone(self.reset)
        for org, owner in ((self.org_a, self.owner_a), (self.org_b, self.owner_b)):
            outbox = self.store.list_outbox(org, owner)
            if outbox.empty:
                continue
            self.assertNotIn("password_reset", set(outbox["kind"]))
            bodies = " ".join(outbox["body"].astype(str))
            self.assertNotIn(self.reset["token"], bodies)
            self.assertNotIn(self.reset["reset_url"], bodies)

    def test_outbox_requires_admin_role_in_that_organization(self):
        with self.assertRaises(PermissionError):
            self.store.list_outbox(self.org_a, self.owner_b)
        user_id, _ = self.store.accept_invitation(
            self.invite_a["token"],
            display_name="Scientist A",
            password="SecurePass123",
        )
        with self.assertRaises(PermissionError):
            self.store.list_outbox(self.org_a, user_id)

    def test_delivery_worker_still_sees_all_queued_messages(self):
        queued = self.store.list_notifications(status="queued")
        kinds = set(queued["kind"])
        self.assertIn("password_reset", kinds)
        self.assertGreaterEqual(len(queued), 3)


if __name__ == "__main__":
    unittest.main()
