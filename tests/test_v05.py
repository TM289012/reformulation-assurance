from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import tempfile
import unittest

import pandas as pd

from closed_loop import ensure_v04_config
from dossier import compute_evidence_hash, generate_dossier
from ingestion import import_readiness_report, load_table, workbook_preview
from product_store import ProductStore

ROOT = Path(__file__).resolve().parents[1]


def config_for(data: pd.DataFrame) -> dict:
    mixture_columns = [c for c in data.columns if c.startswith("ingredient_")]
    baseline = data[data["status"] == "completed"].iloc[0][
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
        "candidate_pool_size": 300,
        "min_distance": 0.05,
    })


class V05SecurityAndTenantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProductStore(Path(self.tmp.name) / "v05.db")
        self.owner_a, self.org_a = self.store.register_owner(
            email="owner-a@example.com", display_name="Owner A",
            password="SecurePass123", organization_name="Lab A",
        )
        self.owner_b, self.org_b = self.store.register_owner(
            email="owner-b@example.com", display_name="Owner B",
            password="AnotherPass123", organization_name="Lab B",
        )
        self.data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        self.config = config_for(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_authentication_and_wrong_password(self):
        user = self.store.authenticate("OWNER-A@example.com", "SecurePass123")
        self.assertEqual(user["id"], self.owner_a)
        self.assertIsNone(self.store.authenticate("owner-a@example.com", "wrong"))

    def test_organization_project_isolation(self):
        project_a = self.store.create_project(
            "Project A", self.config, organization_id=self.org_a, created_by_user_id=self.owner_a
        )
        self.store.create_project(
            "Project B", self.config, organization_id=self.org_b, created_by_user_id=self.owner_b
        )
        projects_a = self.store.list_projects(self.org_a)
        projects_b = self.store.list_projects(self.org_b)
        self.assertEqual(projects_a["name"].tolist(), ["Project A"])
        self.assertEqual(projects_b["name"].tolist(), ["Project B"])
        with self.assertRaises(PermissionError):
            self.store.require_project_access(self.owner_b, project_a)

    def test_roles_and_reauthentication_at_signature(self):
        project_id = self.store.create_project(
            "Approval Test", self.config, organization_id=self.org_a, created_by_user_id=self.owner_a
        )
        self.store.import_history(project_id, self.data.head(20))
        approver = self.store.create_member(
            self.org_a, email="approver@example.com", display_name="Quality Approver",
            password="ApproverPass123", role="approver", actor_user_id=self.owner_a,
        )
        evidence_hash = compute_evidence_hash(self.store, project_id)
        with self.assertRaises(PermissionError):
            self.store.sign_approval(
                project_id, stage="confirmation", signer_user_id=approver,
                typed_name="Quality Approver", password="wrong",
                signature_meaning="I approve progression.", evidence_hash=evidence_hash,
            )
        approval_id = self.store.sign_approval(
            project_id, stage="confirmation", signer_user_id=approver,
            typed_name="Quality Approver", password="ApproverPass123",
            signature_meaning="I approve progression.", evidence_hash=evidence_hash,
        )
        approvals = self.store.list_approvals(project_id)
        self.assertEqual(approvals.iloc[0]["id"], approval_id)
        updated = dict(self.config)
        updated["n_recommendations"] = 4
        self.store.update_project_config(project_id, updated)
        self.assertNotEqual(compute_evidence_hash(self.store, project_id), evidence_hash)


class V05IngestionAndDossierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProductStore(Path(self.tmp.name) / "v05.db")
        self.owner, self.org = self.store.register_owner(
            email="owner@example.com", display_name="Dossier Owner",
            password="SecurePass123", organization_name="Dossier Lab",
        )
        self.data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
        self.config = config_for(self.data)
        self.project_id = self.store.create_project(
            "Dossier Project", self.config, description="Qualification evidence test",
            organization_id=self.org, created_by_user_id=self.owner,
        )
        self.store.import_history(self.project_id, self.data.head(30))

    def tearDown(self):
        self.tmp.cleanup()

    def test_multisheet_excel_ingestion(self):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            self.data.head(5).to_excel(writer, sheet_name="Experiments", index=False)
            pd.DataFrame({"note": ["metadata"]}).to_excel(writer, sheet_name="Notes", index=False)
        raw = buffer.getvalue()
        preview = workbook_preview(raw, "campaign.xlsx")
        self.assertEqual(preview.sheets, ["Experiments", "Notes"])
        loaded = load_table(raw, "campaign.xlsx", sheet_name="Experiments")
        self.assertEqual(len(loaded), 5)
        readiness = import_readiness_report(loaded)
        self.assertEqual(set(readiness["column"]), set(loaded.columns))

    def test_dossier_contains_manifest_evidence_and_valid_checksums(self):
        evidence_hash = compute_evidence_hash(self.store, self.project_id)
        self.store.sign_approval(
            self.project_id, stage="discovery", signer_user_id=self.owner,
            typed_name="Dossier Owner", password="SecurePass123",
            signature_meaning="I reviewed the available discovery evidence.",
            evidence_hash=evidence_hash,
        )
        payload, manifest = generate_dossier(
            self.store, self.project_id, generated_by_user_id=self.owner
        )
        self.assertEqual(manifest["scientific_evidence_sha256"], evidence_hash)
        with ZipFile(BytesIO(payload)) as archive:
            names = set(archive.namelist())
            required = {
                "qualification_dossier.html", "manifest.json", "scientific_evidence.json",
                "experiments.csv", "approvals.csv", "audit_trail.csv", "SHA256SUMS.txt",
            }
            self.assertTrue(required.issubset(names))
            checksum_lines = archive.read("SHA256SUMS.txt").decode("utf-8").strip().splitlines()
            for line in checksum_lines:
                digest, filename = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(archive.read(filename)).hexdigest(), digest)
        dossiers = self.store.list_dossiers(self.project_id)
        self.assertEqual(int(dossiers.iloc[0]["version"]), 1)


if __name__ == "__main__":
    unittest.main()
