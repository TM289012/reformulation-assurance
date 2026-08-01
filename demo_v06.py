"""End-to-end v0.6 pilot collaboration and operations demonstration."""
from __future__ import annotations

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import tempfile

import pandas as pd

from artifact_vault import ArtifactVault
from backup_service import create_backup
from closed_loop import ensure_v04_config
from dossier import compute_evidence_hash, generate_dossier
from pilot_store import PilotStore
from postgres_migration import create_postgres_migration_bundle

ROOT = Path(__file__).resolve().parent


def build_config(data: pd.DataFrame) -> dict:
    mixture = [column for column in data.columns if column.startswith("ingredient_")]
    baseline = data[data["status"] == "completed"].iloc[0][
        [*mixture, "mix_temperature_c", "mix_time_min", "supplier_family"]
    ].to_dict()
    return ensure_v04_config({
        "mixture_columns": mixture,
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
        "ingredient_costs": {column: 1.0 for column in mixture},
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 300,
        "min_distance": 0.05,
    })


def run_demo(output_directory: str | Path | None = None) -> dict:
    output_directory = Path(output_directory or ROOT)
    data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        store = PilotStore(temp_root / "demo_v06.db")
        vault = ArtifactVault(temp_root / "artifacts")
        owner, organization = store.register_owner(
            email="owner@example.com",
            display_name="Demo Owner",
            password="SecurePass123",
            organization_name="Demo Formulation Lab",
        )
        invitation = store.create_invitation(
            organization,
            email="approver@example.com",
            role="approver",
            actor_user_id=owner,
            base_url="https://pilot.example",
        )
        approver, _ = store.accept_invitation(
            invitation["token"],
            display_name="Quality Approver",
            password="ApproverPass123",
        )
        project = store.create_project(
            "Coating Qualification Pilot",
            build_config(data),
            description="Demonstrate audited collaboration and two-role approval.",
            organization_id=organization,
            created_by_user_id=owner,
        )
        store.import_history(project, data.head(36))
        store.add_comment(
            project,
            author_user_id=owner,
            body="Quality should confirm that the current evidence supports the confirmation gate.",
        )
        store.create_assignment(
            project,
            title="Review confirmation evidence",
            assignee_user_id=approver,
            created_by_user_id=owner,
            priority="high",
        )
        policy = store.create_approval_policy(
            project,
            stage="confirmation",
            name="Owner plus quality approval",
            requirements=[{"role": "owner", "count": 1}, {"role": "approver", "count": 1}],
            actor_user_id=owner,
        )
        evidence_hash = compute_evidence_hash(store, project)
        store.sign_approval(
            project,
            stage="confirmation",
            signer_user_id=owner,
            typed_name="Demo Owner",
            password="SecurePass123",
            signature_meaning="Technical evidence reviewed.",
            evidence_hash=evidence_hash,
            policy_id=policy,
        )
        store.sign_approval(
            project,
            stage="confirmation",
            signer_user_id=approver,
            typed_name="Quality Approver",
            password="ApproverPass123",
            signature_meaning="Quality evidence reviewed.",
            evidence_hash=evidence_hash,
            policy_id=policy,
        )
        policy_status = store.approval_policy_status(project, evidence_hash)
        dossier, manifest = generate_dossier(store, project, generated_by_user_id=owner)
        artifact_id = vault.store_project_artifact(
            store,
            project,
            created_by_user_id=owner,
            payload=dossier,
            filename="sample_v06_qualification_dossier.zip",
            artifact_type="qualification_dossier",
            content_type="application/zip",
            metadata=manifest,
        )
        recovered, _ = vault.retrieve_project_artifact(store, artifact_id, owner)
        backup_id, _ = create_backup(
            store,
            vault,
            organization_id=organization,
            created_by_user_id=owner,
        )
        migration, migration_manifest = create_postgres_migration_bundle(store.database_path)

        dossier_path = output_directory / "sample_v06_qualification_dossier.zip"
        migration_path = output_directory / "sample_v06_postgres_migration.zip"
        dossier_path.write_bytes(recovered)
        migration_path.write_bytes(migration)
        return {
            "invitation_status": store.list_invitations(organization, owner).iloc[0]["status"],
            "comments": len(store.list_comments(project, owner)),
            "assignments": len(store.list_assignments(project, owner)),
            "policy_complete": bool(policy_status.iloc[0]["complete"]),
            "artifact_id": artifact_id,
            "backup_id": backup_id,
            "backup_status": store.list_backups(organization_id=organization).iloc[0]["status"],
            "dossier_files": len(ZipFile(BytesIO(recovered)).namelist()),
            "migration_tables": len(migration_manifest["table_counts"]),
            "dossier_path": str(dossier_path),
            "migration_path": str(migration_path),
        }


def main() -> None:
    result = run_demo()
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
