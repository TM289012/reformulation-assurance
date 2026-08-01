"""End-to-end v0.5 productization demonstration."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import tempfile

import pandas as pd

from closed_loop import ensure_v04_config
from dossier import compute_evidence_hash, generate_dossier
from ingestion import load_table, workbook_preview
from product_store import ProductStore

ROOT = Path(__file__).resolve().parent


def build_config(data: pd.DataFrame) -> dict:
    mixture = [c for c in data.columns if c.startswith("ingredient_")]
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
            "ingredient_resin": [34.0, 62.0], "ingredient_crosslinker": [8.0, 20.0],
            "ingredient_solvent": [12.0, 35.0], "ingredient_legacy_plasticizer": [0.0, 0.0],
            "ingredient_substitute_a": [0.0, 18.0], "ingredient_substitute_b": [0.0, 15.0],
        },
        "process_bounds": {"mix_temperature_c": [48.0, 72.0], "mix_time_min": [18.0, 42.0]},
        "category_values": {"supplier_family": ["A", "B", "C"]},
        "mixture_total": 100.0,
        "ingredient_to_remove": "ingredient_legacy_plasticizer",
        "baseline": baseline,
        "ingredient_costs": {c: 1.0 for c in mixture},
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 300,
        "min_distance": 0.05,
    })


def main() -> None:
    data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Experiments", index=False)
        pd.DataFrame({"owner": ["Demo Lab"]}).to_excel(writer, sheet_name="Metadata", index=False)
    preview = workbook_preview(workbook.getvalue(), "demo.xlsx")
    imported = load_table(workbook.getvalue(), "demo.xlsx", sheet_name="Experiments")

    with tempfile.TemporaryDirectory() as temp:
        store = ProductStore(Path(temp) / "demo.db")
        owner, organization = store.register_owner(
            email="owner@example.com", display_name="Demo Owner",
            password="SecurePass123", organization_name="Demo Lab",
        )
        project = store.create_project(
            "Coating Qualification", build_config(imported),
            organization_id=organization, created_by_user_id=owner,
        )
        store.import_history(project, imported)
        evidence_hash = compute_evidence_hash(store, project)
        store.sign_approval(
            project, stage="discovery", signer_user_id=owner,
            typed_name="Demo Owner", password="SecurePass123",
            signature_meaning="I reviewed the discovery evidence.", evidence_hash=evidence_hash,
        )
        dossier, manifest = generate_dossier(store, project, generated_by_user_id=owner)
        with ZipFile(BytesIO(dossier)) as archive:
            files = len(archive.namelist())
        print(f"Excel sheets: {preview.sheets}")
        print(f"Imported rows: {len(imported)}")
        print(f"Tenant projects: {len(store.list_projects(organization))}")
        print(f"Evidence hash: {evidence_hash[:16]}…")
        print(f"Dossier version: {manifest['version']}")
        print(f"Dossier files: {files}")


if __name__ == "__main__":
    main()
