from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import tempfile
import unittest

import pandas as pd

from artifact_vault import ArtifactVault
from backup_service import create_backup, restore_backup_payload
from dossier import compute_evidence_hash, generate_dossier
from pilot_store import PilotStore
from postgres_migration import create_postgres_migration_bundle
from tests.test_v05 import config_for

ROOT = Path(__file__).resolve().parents[1]


class V06PilotFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = PilotStore(self.root / "pilot.db")
        self.owner, self.org = self.store.register_owner(
            email="owner@example.com",
            display_name="Pilot Owner",
            password="SecurePass123",
            organization_name="Pilot Lab",
        )
        self.data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        self.config = config_for(self.data)
        self.project = self.store.create_project(
            "Pilot Reformulation",
            self.config,
            organization_id=self.org,
            created_by_user_id=self.owner,
        )
        self.store.import_history(self.project, self.data.head(24))

    def tearDown(self):
        self.tmp.cleanup()


class V06IdentityTests(V06PilotFixture):
    def test_invitation_acceptance_and_outbox(self):
        invitation = self.store.create_invitation(
            self.org,
            email="scientist@example.com",
            role="scientist",
            actor_user_id=self.owner,
            base_url="https://pilot.example",
        )
        self.assertIn("?invite=", invitation["invite_url"])
        queued = self.store.list_notifications(status="queued")
        self.assertEqual(len(queued), 1)
        user_id, organization_id = self.store.accept_invitation(
            invitation["token"],
            display_name="Invited Scientist",
            password="ScientistPass123",
        )
        self.assertEqual(organization_id, self.org)
        self.assertEqual(self.store.role_for(user_id, self.org), "scientist")
        with self.assertRaises(ValueError):
            self.store.accept_invitation(
                invitation["token"],
                display_name="Invited Scientist",
                password="ScientistPass123",
            )

    def test_password_reset_is_single_use(self):
        reset = self.store.request_password_reset("owner@example.com", base_url="https://pilot.example")
        self.assertIsNotNone(reset)
        self.store.reset_password(reset["token"], "NewSecurePass456")
        self.assertIsNotNone(self.store.authenticate("owner@example.com", "NewSecurePass456"))
        self.assertIsNone(self.store.authenticate("owner@example.com", "SecurePass123"))
        with self.assertRaises(ValueError):
            self.store.reset_password(reset["token"], "AnotherPass789")


class V06CollaborationAndApprovalTests(V06PilotFixture):
    def setUp(self):
        super().setUp()
        self.scientist = self.store.create_member(
            self.org,
            email="scientist@example.com",
            display_name="Project Scientist",
            password="ScientistPass123",
            role="scientist",
            actor_user_id=self.owner,
        )
        self.approver = self.store.create_member(
            self.org,
            email="approver@example.com",
            display_name="Quality Approver",
            password="ApproverPass123",
            role="approver",
            actor_user_id=self.owner,
        )

    def test_comments_and_assignments_are_audited(self):
        comment = self.store.add_comment(
            self.project,
            author_user_id=self.scientist,
            body="Please confirm supplier-lot evidence.",
        )
        assignment = self.store.create_assignment(
            self.project,
            title="Run supplier-lot replicate",
            assignee_user_id=self.scientist,
            created_by_user_id=self.owner,
            priority="high",
        )
        self.store.update_assignment(assignment, self.scientist, status="in_progress")
        self.store.resolve_comment(comment, self.scientist)
        comments = self.store.list_comments(self.project, self.owner)
        tasks = self.store.list_assignments(self.project, self.owner)
        self.assertIsNotNone(comments.iloc[0]["resolved_at"])
        self.assertEqual(tasks.iloc[0]["status"], "in_progress")
        events = set(self.store.audit_log(self.project)["event_type"])
        self.assertTrue({"comment_added", "comment_resolved", "assignment_created", "assignment_updated"}.issubset(events))

    def test_multi_signer_policy_requires_distinct_roles(self):
        policy = self.store.create_approval_policy(
            self.project,
            stage="confirmation",
            name="Technical and quality review",
            requirements=[{"role": "owner", "count": 1}, {"role": "approver", "count": 1}],
            actor_user_id=self.owner,
        )
        evidence = compute_evidence_hash(self.store, self.project)
        self.store.sign_approval(
            self.project,
            stage="confirmation",
            signer_user_id=self.owner,
            typed_name="Pilot Owner",
            password="SecurePass123",
            signature_meaning="Technical evidence reviewed.",
            evidence_hash=evidence,
            policy_id=policy,
        )
        status = self.store.approval_policy_status(self.project, evidence)
        self.assertFalse(bool(status.iloc[0]["complete"]))
        self.store.sign_approval(
            self.project,
            stage="confirmation",
            signer_user_id=self.approver,
            typed_name="Quality Approver",
            password="ApproverPass123",
            signature_meaning="Quality evidence reviewed.",
            evidence_hash=evidence,
            policy_id=policy,
        )
        status = self.store.approval_policy_status(self.project, evidence)
        self.assertTrue(bool(status.iloc[0]["complete"]))
        with self.assertRaises(ValueError):
            self.store.sign_approval(
                self.project,
                stage="confirmation",
                signer_user_id=self.approver,
                typed_name="Quality Approver",
                password="ApproverPass123",
                signature_meaning="Duplicate.",
                evidence_hash=evidence,
                policy_id=policy,
            )


class V06ArtifactBackupMigrationTests(V06PilotFixture):
    def setUp(self):
        super().setUp()
        self.vault = ArtifactVault(self.root / "artifacts")

    def test_encrypted_dossier_round_trip(self):
        dossier, manifest = generate_dossier(self.store, self.project, generated_by_user_id=self.owner)
        artifact_id = self.vault.store_project_artifact(
            self.store,
            self.project,
            created_by_user_id=self.owner,
            payload=dossier,
            filename="qualification.zip",
            artifact_type="qualification_dossier",
            content_type="application/zip",
            metadata={"version": manifest["version"]},
        )
        recovered, record = self.vault.retrieve_project_artifact(self.store, artifact_id, self.owner)
        self.assertEqual(recovered, dossier)
        self.assertEqual(hashlib.sha256(recovered).hexdigest(), record["plaintext_sha256"])
        self.assertNotEqual(Path(record["storage_path"]).read_bytes(), dossier)

    def test_verified_backup_and_restore(self):
        backup_id, _ = create_backup(
            self.store,
            self.vault,
            organization_id=self.org,
            created_by_user_id=self.owner,
        )
        records = self.store.list_backups(organization_id=self.org)
        record = records[records["id"] == backup_id].iloc[0].to_dict()
        self.assertEqual(record["status"], "verified")
        restored = restore_backup_payload(self.vault, record, self.root / "restored.db")
        restored_store = PilotStore(restored)
        self.assertEqual(len(restored_store.list_projects(self.org)), 1)

    def test_postgres_migration_bundle_is_checksummed(self):
        payload, manifest = create_postgres_migration_bundle(self.store.database_path)
        self.assertGreater(manifest["table_counts"]["projects"], 0)
        with ZipFile(BytesIO(payload)) as archive:
            lines = archive.read("SHA256SUMS.txt").decode().strip().splitlines()
            for line in lines:
                digest, filename = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(archive.read(filename)).hexdigest(), digest)
            exported_manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(exported_manifest["format_version"], 1)


if __name__ == "__main__":
    unittest.main()
