"""Demo-mode seeding for public sandbox deployments (e.g. Streamlit Community Cloud).

When ``REFORMULATION_DEMO_MODE`` is enabled, an empty database is seeded with a
shared demo workspace and the cosmetics emulsifier-swap project preloaded, so a
visitor can explore recommendations, qualification stages, approvals, and the
dossier export without installing anything or entering real data.

The sandbox is shared and periodically wiped. Never enable demo mode on a
deployment that holds real formulation data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from assurance_v4 import default_variation_config
from closed_loop import ensure_v04_config
from pilot_store import PilotStore

ROOT = Path(__file__).resolve().parent

DEMO_OWNER_EMAIL = "demo@example.com"
DEMO_OWNER_PASSWORD = "try-the-demo-2026"
DEMO_ORGANIZATION = "Public Demo Lab"
DEMO_CSV = ROOT / "demo_cosmetics_emulsifier_swap.csv"


def cosmetics_config(data: pd.DataFrame) -> dict:
    """Project configuration for the cosmetics emulsifier-swap demo dataset."""
    mixture_columns = [column for column in data.columns if column.startswith("ingredient_")]
    completed = data[data["status"] == "completed"]
    legacy = completed[completed["ingredient_legacy_peg_emulsifier"] > 1.0]
    baseline_row = (legacy if not legacy.empty else completed).iloc[0]
    baseline = baseline_row[
        [*mixture_columns, "emulsification_temp_c", "homogenization_min", "supplier_family"]
    ].to_dict()
    config = ensure_v04_config({
        "mixture_columns": mixture_columns,
        "process_columns": ["emulsification_temp_c", "homogenization_min"],
        "categorical_columns": ["supplier_family"],
        "response_specs": [
            {"response": "viscosity_cp", "minimum": 8000.0, "maximum": 14000.0, "weight": 1.0},
            {"response": "ph", "minimum": 5.0, "maximum": 5.6, "weight": 1.0},
            {"response": "stability_score", "minimum": 8.5, "maximum": None, "weight": 1.5},
            {"response": "spreadability", "minimum": 7.5, "maximum": None, "weight": 1.0},
        ],
        "mixture_bounds": {
            "ingredient_water": [66.0, 78.0],
            "ingredient_oil_phase": [13.0, 21.0],
            "ingredient_legacy_peg_emulsifier": [0.0, 0.0],
            "ingredient_substitute_polyglyceryl": [0.0, 5.0],
            "ingredient_substitute_glucoside": [0.0, 2.6],
            "ingredient_glycerin": [3.0, 8.5],
            "ingredient_thickener": [0.35, 1.55],
        },
        "process_bounds": {
            "emulsification_temp_c": [65.0, 80.0],
            "homogenization_min": [3.0, 12.0],
        },
        "category_values": {"supplier_family": ["A", "B", "C"]},
        "mixture_total": 100.0,
        "ingredient_to_remove": "ingredient_legacy_peg_emulsifier",
        "baseline": baseline,
        "ingredient_costs": {
            "ingredient_water": 0.05,
            "ingredient_oil_phase": 2.20,
            "ingredient_legacy_peg_emulsifier": 3.10,
            "ingredient_substitute_polyglyceryl": 4.60,
            "ingredient_substitute_glucoside": 3.90,
            "ingredient_glycerin": 1.20,
            "ingredient_thickener": 6.50,
        },
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 2500,
        "min_distance": 0.06,
    })
    config["manufacturing_variation"] = default_variation_config(config)
    return config


def seed_demo(store: PilotStore) -> str | None:
    """Create the shared demo workspace if the database is empty.

    Idempotent: does nothing when any user already exists, so a restarted
    deployment keeps whatever state visitors have built until the next wipe.
    Returns the new project id when seeding ran, else None.
    """
    if store.has_users():
        return None
    user_id, organization_id = store.register_owner(
        email=DEMO_OWNER_EMAIL,
        display_name="Demo Explorer",
        password=DEMO_OWNER_PASSWORD,
        organization_name=DEMO_ORGANIZATION,
    )
    data = pd.read_csv(DEMO_CSV)
    project_id = store.create_project(
        "Cosmetics: PEG emulsifier replacement (demo)",
        cosmetics_config(data),
        description=(
            "Replace a discontinued PEG emulsifier in an oil-in-water lotion. "
            "88 historical lots including 7 failed emulsions. Shared sandbox: "
            "run the model, create batches, sign approvals — data resets periodically."
        ),
        source_filename=DEMO_CSV.name,
        organization_id=organization_id,
        created_by_user_id=user_id,
    )
    store.import_history(project_id, data)
    return project_id
