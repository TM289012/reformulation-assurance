"""The example reformulation projects that ship with the app.

Three synthetic histories, one per industry the tool is used in, each with the project
configuration (mixture bounds, process window, specifications, costs) that turns the CSV
into a working project:

* ``coatings``: replace a legacy plasticizer in a coating (the original v0.6 walkthrough).
* ``cosmetics``: replace a discontinued PEG emulsifier in an oil-in-water lotion.
* ``food``: replace FD&C Red No. 3 in a strawberry gummy with a natural colorant.

``app.py`` offers them under New project, Start with demo; ``demo_seed.py`` preloads the
public sandbox with the cosmetics and food projects. Every dataset is synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from assurance_v4 import default_variation_config
from closed_loop import ensure_v04_config

ROOT = Path(__file__).resolve().parent


def _finish(base: dict[str, Any]) -> dict[str, Any]:
    config = ensure_v04_config(base)
    config["manufacturing_variation"] = default_variation_config(config)
    return config


def coatings_config(data: pd.DataFrame) -> dict[str, Any]:
    """Project configuration for ``demo_coatings_reformulation.csv``."""
    mixture_columns = [column for column in data.columns if column.startswith("ingredient_")]
    completed = data[data["status"] == "completed"]
    baseline = completed.iloc[0][
        [*mixture_columns, "mix_temperature_c", "mix_time_min", "supplier_family"]
    ].to_dict()
    return _finish({
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
        "process_bounds": {
            "mix_temperature_c": [48.0, 72.0],
            "mix_time_min": [18.0, 42.0],
        },
        "category_values": {"supplier_family": ["A", "B", "C"]},
        "mixture_total": 100.0,
        "ingredient_to_remove": "ingredient_legacy_plasticizer",
        "baseline": baseline,
        "ingredient_costs": {
            "ingredient_resin": 2.50,
            "ingredient_crosslinker": 4.80,
            "ingredient_solvent": 0.90,
            "ingredient_legacy_plasticizer": 1.30,
            "ingredient_substitute_a": 1.85,
            "ingredient_substitute_b": 2.20,
        },
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 2500,
        "min_distance": 0.06,
    })


def cosmetics_config(data: pd.DataFrame) -> dict[str, Any]:
    """Project configuration for ``demo_cosmetics_emulsifier_swap.csv``."""
    mixture_columns = [column for column in data.columns if column.startswith("ingredient_")]
    completed = data[data["status"] == "completed"]
    legacy = completed[completed["ingredient_legacy_peg_emulsifier"] > 1.0]
    baseline_row = (legacy if not legacy.empty else completed).iloc[0]
    baseline = baseline_row[
        [*mixture_columns, "emulsification_temp_c", "homogenization_min", "supplier_family"]
    ].to_dict()
    return _finish({
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


def food_color_config(data: pd.DataFrame) -> dict[str, Any]:
    """Project configuration for ``demo_food_color_replacement.csv``.

    The specifications are the ones a brand would write for a drop-in color match:
    within a color difference of 3.0 of the Red No. 3 standard, at least as red as
    a* 38, keeping 85 percent of the color through the four-week storage test, and a
    panel flavor score of at least 7.5 so an earthy beet note cannot be traded for color.
    """
    mixture_columns = [column for column in data.columns if column.startswith("ingredient_")]
    completed = data[data["status"] == "completed"]
    legacy = completed[completed["ingredient_legacy_red3"] > 0.01]
    baseline_row = (legacy if not legacy.empty else completed).iloc[0]
    baseline = baseline_row[
        [*mixture_columns, "cook_temp_c", "hold_time_min", "supplier_family"]
    ].to_dict()
    return _finish({
        "mixture_columns": mixture_columns,
        "process_columns": ["cook_temp_c", "hold_time_min"],
        "categorical_columns": ["supplier_family"],
        "response_specs": [
            {"response": "delta_e_vs_standard", "minimum": None, "maximum": 3.0, "weight": 1.5},
            {"response": "a_star", "minimum": 38.0, "maximum": None, "weight": 1.0},
            {"response": "color_retention_pct", "minimum": 85.0, "maximum": None, "weight": 1.5},
            {"response": "flavor_score", "minimum": 7.5, "maximum": None, "weight": 1.0},
        ],
        "mixture_bounds": {
            "ingredient_glucose_syrup": [37.0, 47.0],
            "ingredient_sugar": [27.0, 37.0],
            "ingredient_water": [9.0, 25.0],
            "ingredient_gelatin": [6.0, 9.0],
            "ingredient_citric_acid": [0.8, 1.7],
            "ingredient_legacy_red3": [0.0, 0.0],
            "ingredient_substitute_black_carrot": [0.0, 1.4],
            "ingredient_substitute_beet": [0.0, 2.2],
        },
        "process_bounds": {
            "cook_temp_c": [106.0, 128.0],
            "hold_time_min": [2.0, 12.0],
        },
        "category_values": {"supplier_family": ["A", "B", "C"]},
        "mixture_total": 100.0,
        "ingredient_to_remove": "ingredient_legacy_red3",
        "baseline": baseline,
        "ingredient_costs": {
            "ingredient_glucose_syrup": 0.70,
            "ingredient_sugar": 0.75,
            "ingredient_water": 0.01,
            "ingredient_gelatin": 9.00,
            "ingredient_citric_acid": 1.60,
            "ingredient_legacy_red3": 55.00,
            "ingredient_substitute_black_carrot": 42.00,
            "ingredient_substitute_beet": 12.00,
        },
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 2500,
        "min_distance": 0.06,
    })


@dataclass(frozen=True)
class DemoDataset:
    key: str
    title: str
    project_name: str
    description: str
    summary: str
    filename: str
    build_config: Callable[[pd.DataFrame], dict[str, Any]]

    @property
    def path(self) -> Path:
        return ROOT / self.filename

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


DEMO_DATASETS: tuple[DemoDataset, ...] = (
    DemoDataset(
        key="coatings",
        title="Coatings: legacy plasticizer replacement",
        project_name="Coating Plasticizer Replacement v0.6",
        description=(
            "Replace the legacy plasticizer and prove repeatability, robustness, approval, "
            "and qualification evidence."
        ),
        summary=(
            "A coating that loses its plasticizer. 60 historical trials, completed and failed, "
            "with adhesion, viscosity, dry time and gloss specifications. The original walkthrough."
        ),
        filename="demo_coatings_reformulation.csv",
        build_config=coatings_config,
    ),
    DemoDataset(
        key="cosmetics",
        title="Cosmetics: PEG emulsifier replacement",
        project_name="Cosmetics: PEG emulsifier replacement (demo)",
        description=(
            "Replace a discontinued PEG emulsifier in an oil-in-water lotion. "
            "88 historical lots including 7 failed emulsions."
        ),
        summary=(
            "An oil-in-water lotion whose PEG emulsifier is discontinued. 88 lots including "
            "7 failed emulsions, with viscosity, pH, stability and spreadability specifications."
        ),
        filename="demo_cosmetics_emulsifier_swap.csv",
        build_config=cosmetics_config,
    ),
    DemoDataset(
        key="food",
        title="Food: FD&C Red No. 3 replacement",
        project_name="Food: Red No. 3 replacement in a strawberry gummy (demo)",
        description=(
            "Replace FD&C Red No. 3 (food use ends 15 January 2027) with a natural colorant in a "
            "strawberry gummy. 90 historical lots: 18 legacy lots that set the color standard, "
            "72 trials with black carrot anthocyanin and beet concentrate, 8 of them failed."
        ),
        summary=(
            "A strawberry gummy colored with Red No. 3, which loses its food use on 15 January 2027. "
            "90 lots: 18 legacy lots that define the color standard and 72 trials of black carrot "
            "anthocyanin and beet concentrate at different cook temperatures. Specifications on color "
            "difference, redness, color retention through storage, and flavor."
        ),
        filename="demo_food_color_replacement.csv",
        build_config=food_color_config,
    ),
)

DEMO_BY_KEY: dict[str, DemoDataset] = {dataset.key: dataset for dataset in DEMO_DATASETS}

# What the public sandbox preloads, in creation order.
SANDBOX_KEYS: tuple[str, ...] = ("cosmetics", "food")


def demo_dataset(key: str) -> DemoDataset:
    try:
        return DEMO_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown demo dataset: {key!r}; choose from {sorted(DEMO_BY_KEY)}") from exc


def create_demo_project(store: Any, key: str, *, organization_id: str | None, created_by_user_id: str | None) -> str:
    """Create one of the built-in demo projects in ``store`` and import its history."""
    dataset = demo_dataset(key)
    data = dataset.load()
    project_id = store.create_project(
        dataset.project_name,
        dataset.build_config(data),
        description=dataset.description,
        source_filename=dataset.filename,
        organization_id=organization_id,
        created_by_user_id=created_by_user_id,
    )
    store.import_history(project_id, data)
    return project_id
