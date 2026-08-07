"""Tests for signed evidence snapshots (v0.8.0).

A signature must preserve the exact evidence it covered: the canonical-JSON
snapshot is stored beside the SHA-256 hash, survives later edits to the live
data, and always re-hashes to the hash recorded at signing.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_seed import DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD, seed_demo  # noqa: E402
from dossier import compute_evidence_hash, evidence_snapshot_and_hash, generate_dossier  # noqa: E402
from pilot_store import PilotStore  # noqa: E402

SIGNATURE_MEANING = "I reviewed the evidence for this stage and approve progression."


class EvidenceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tempdir.name) / "workspace.db")
        self.project_id = seed_demo(self.store)
        self.user = self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)

    def tearDown(self):
        self.tempdir.cleanup()

    def _sign(self, stage: str, snapshot: str | None, evidence_hash: str) -> str:
        return self.store.sign_approval(
            self.project_id,
            stage=stage,
            signer_user_id=self.user["id"],
            typed_name=self.user["display_name"],
            password=DEMO_OWNER_PASSWORD,
            signature_meaning=SIGNATURE_MEANING,
            evidence_hash=evidence_hash,
            evidence_snapshot=snapshot,
        )

    def test_snapshot_survives_evidence_change_and_reverifies(self):
        snapshot, evidence_hash = evidence_snapshot_and_hash(self.store, self.project_id)
        self.assertEqual(sha256(snapshot.encode("utf-8")).hexdigest(), evidence_hash)
        self._sign("screening", snapshot, evidence_hash)

        # Mutate the live evidence after signing.
        config = dict(self.store.get_project(self.project_id)["config"])
        config["snapshot_test_marker"] = "changed after signing"
        self.store.update_project_config(self.project_id, config)

        approvals = self.store.list_approvals(self.project_id)
        row = approvals.iloc[0]
        stored = row["evidence_snapshot"]
        self.assertIsInstance(stored, str)
        self.assertEqual(stored, snapshot)  # frozen copy is untouched
        self.assertEqual(sha256(stored.encode("utf-8")).hexdigest(), row["evidence_hash"])
        # Stale detection still works: live evidence no longer matches the signed hash.
        self.assertNotEqual(compute_evidence_hash(self.store, self.project_id), row["evidence_hash"])

    def test_signing_without_snapshot_still_works(self):
        snapshot, evidence_hash = evidence_snapshot_and_hash(self.store, self.project_id)
        self._sign("robustness", None, evidence_hash)
        approvals = self.store.list_approvals(self.project_id)
        row = approvals.iloc[0]
        stored = row["evidence_snapshot"]
        self.assertFalse(isinstance(stored, str) and stored)

    def test_migration_adds_column_to_pre_v080_database(self):
        db_path = Path(self.tempdir.name) / "legacy.db"
        legacy = PilotStore(db_path)
        with legacy.connection() as con:
            con.execute("ALTER TABLE approvals DROP COLUMN evidence_snapshot")
            columns = {row[1] for row in con.execute("PRAGMA table_info(approvals)").fetchall()}
            self.assertNotIn("evidence_snapshot", columns)
        reopened = PilotStore(db_path)
        with reopened.connection() as con:
            columns = {row[1] for row in con.execute("PRAGMA table_info(approvals)").fetchall()}
        self.assertIn("evidence_snapshot", columns)

    def test_dossier_zip_contains_signed_snapshots(self):
        snapshot, evidence_hash = evidence_snapshot_and_hash(self.store, self.project_id)
        self._sign("screening", snapshot, evidence_hash)
        dossier_bytes, _manifest = generate_dossier(
            self.store, self.project_id, generated_by_user_id=self.user["id"]
        )
        with ZipFile(BytesIO(dossier_bytes)) as archive:
            names = archive.namelist()
            snapshot_names = [name for name in names if name.startswith("signed_evidence_snapshots/")]
            self.assertEqual(len(snapshot_names), 1)
            content = archive.read(snapshot_names[0])
            self.assertEqual(sha256(content).hexdigest(), evidence_hash)
            approvals_csv = archive.read("approvals.csv").decode("utf-8")
            self.assertNotIn("evidence_snapshot", approvals_csv)


if __name__ == "__main__":
    unittest.main()
