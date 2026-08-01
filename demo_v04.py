"""Command-line demonstration of Reformulation Assurance v0.4."""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from assurance_v4 import (
    calibration_report,
    result_for_storage,
    simulate_manufacturing_variation,
)
from closed_loop import create_recommendation_batch, ensure_v04_config, qualification_progress
from project_store import ProjectStore


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
        "ingredient_costs": {c: 1.0 for c in mixture},
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 700,
        "min_distance": 0.05,
    })


def main() -> None:
    data = pd.read_csv(ROOT / "demo_coatings_reformulation.csv")
    with tempfile.TemporaryDirectory() as tmp:
        store = ProjectStore(Path(tmp) / "demo.db")
        config = build_config(data)
        project_id = store.create_project("v0.4 demonstration", config)
        store.import_history(project_id, data)
        result, batch_id = create_recommendation_batch(store, project_id, random_state=7)
        store.approve_batch(batch_id)

        experiments = store.list_experiments(project_id, batch_id=batch_id)
        best = experiments.iloc[0]
        store.create_replicates(str(best["id"]), 2)
        replicate_group = store.list_experiments(project_id, batch_id=batch_id)
        replicate_group = replicate_group[replicate_group["replicate_group"] == best["replicate_group"]]
        results = [
            {"adhesion": 8.50, "viscosity_cp": 2290.0, "dry_time_min": 35.0, "gloss": 88.0},
            {"adhesion": 8.54, "viscosity_cp": 2310.0, "dry_time_min": 34.7, "gloss": 88.3},
            {"adhesion": 8.48, "viscosity_cp": 2305.0, "dry_time_min": 35.2, "gloss": 87.8},
        ]
        for (_, experiment), response in zip(replicate_group.iterrows(), results):
            store.update_experiment(
                str(experiment["id"]),
                status="completed",
                responses=response,
                qualification_stage="confirmation",
            )

        candidate_row = store.list_experiments(project_id, batch_id=batch_id).iloc[3]
        candidate = candidate_row.to_dict()
        candidate.update(candidate_row["recommendation"])
        robustness = simulate_manufacturing_variation(
            store.project_dataframe(project_id),
            config=config,
            candidate=candidate,
            variation_std={"mix_temperature_c": 1.0, "mix_time_min": 1.0},
            n_simulations=100,
            random_state=23,
        )
        store.save_robustness_run(
            project_id,
            str(candidate_row["id"]),
            simulation_count=100,
            variation=robustness["variation_std"],
            result=result_for_storage(robustness),
        )

        progress = qualification_progress(store, project_id)
        calibration = calibration_report(store, project_id)
        print("=" * 72)
        print("REFORMULATION ASSURANCE v0.4 — CLOSED ASSURANCE LOOP")
        print("=" * 72)
        print(f"Initial optimizer decision: {result.decision}")
        print(f"Recommended experiments: {len(result.recommendations)}")
        confirmation_row = progress["stage_progress"][
            progress["stage_progress"]["stage"] == "confirmation"
        ].iloc[0]
        print(f"Confirmation replicate gate passed: {bool(confirmation_row['gate_passed'])}")
        print(f"Robust success probability: {robustness['robust_success_probability']:.1%}")
        print(f"Calibration observations: {len(calibration['observations'])}")
        print(f"Qualification score: {progress['score']:.1f}%")
        print(f"Audit events: {len(store.audit_log(project_id))}")


if __name__ == "__main__":
    main()
