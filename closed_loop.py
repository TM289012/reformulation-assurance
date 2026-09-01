"""Closed-loop orchestration for Reformulation Assurance v0.4."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from assurance_v4 import replicate_summary
from project_store import ProjectStore
from reformulation_engine import ReformulationResult, Specification, recommend_reformulations


QUALIFICATION_STAGES = [
    "discovery",
    "confirmation",
    "process_window",
    "raw_material",
    "stability",
    "pilot",
]

STAGE_WEIGHTS = {
    "discovery": 15.0,
    "confirmation": 20.0,
    "process_window": 20.0,
    "raw_material": 15.0,
    "stability": 15.0,
    "pilot": 15.0,
}


def default_qualification_gates(response_specs: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    cv_limits = {item["response"]: 0.08 for item in response_specs}
    return {
        "discovery": {
            "required_completed": 1,
            "required_compliant": 1,
            "minimum_success_rate": 0.50,
            "required_replicate_groups": 0,
            "minimum_replicates_per_group": 1,
            "max_cv_by_response": {},
            "minimum_robust_probability": None,
        },
        "confirmation": {
            "required_completed": 3,
            "required_compliant": 3,
            "minimum_success_rate": 0.80,
            "required_replicate_groups": 1,
            "minimum_replicates_per_group": 3,
            "max_cv_by_response": cv_limits,
            "minimum_robust_probability": None,
        },
        "process_window": {
            "required_completed": 4,
            "required_compliant": 3,
            "minimum_success_rate": 0.75,
            "required_replicate_groups": 0,
            "minimum_replicates_per_group": 1,
            "max_cv_by_response": {},
            "minimum_robust_probability": 0.80,
        },
        "raw_material": {
            "required_completed": 3,
            "required_compliant": 3,
            "minimum_success_rate": 0.80,
            "required_replicate_groups": 0,
            "minimum_replicates_per_group": 1,
            "max_cv_by_response": {},
            "minimum_robust_probability": None,
        },
        "stability": {
            "required_completed": 1,
            "required_compliant": 1,
            "minimum_success_rate": 1.00,
            "required_replicate_groups": 0,
            "minimum_replicates_per_group": 1,
            "max_cv_by_response": {},
            "minimum_robust_probability": None,
        },
        "pilot": {
            "required_completed": 1,
            "required_compliant": 1,
            "minimum_success_rate": 1.00,
            "required_replicate_groups": 0,
            "minimum_replicates_per_group": 1,
            "max_cv_by_response": {},
            "minimum_robust_probability": None,
        },
    }


def ensure_v04_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a v0.4-compatible copy without losing custom project settings."""
    upgraded = deepcopy(dict(config))
    defaults = default_qualification_gates(upgraded.get("response_specs", []))
    existing = upgraded.get("qualification_gates", {})
    merged: dict[str, dict[str, Any]] = {}
    for stage in QUALIFICATION_STAGES:
        merged[stage] = {**defaults[stage], **existing.get(stage, {})}
        merged[stage]["max_cv_by_response"] = {
            **defaults[stage].get("max_cv_by_response", {}),
            **existing.get(stage, {}).get("max_cv_by_response", {}),
        }
    upgraded["qualification_gates"] = merged
    upgraded.setdefault("manufacturing_variation", {})
    upgraded.setdefault("software_version", "0.4")
    return upgraded


def specifications_from_config(config: Mapping[str, Any]) -> list[Specification]:
    return [Specification(**item) for item in config["response_specs"]]


def run_project_model(
    store: ProjectStore,
    project_id: str,
    *,
    random_state: int | None = None,
) -> ReformulationResult:
    project = store.get_project(project_id)
    config = ensure_v04_config(project["config"])
    data = store.project_dataframe(project_id)
    if random_state is None:
        # Deterministic but state-dependent seed: the same project state always
        # reproduces the same candidates (auditable), while every new batch or
        # recorded result refreshes the candidate pool. A fixed seed here would
        # silently re-rank the same finite candidate set forever, so the closed
        # loop would never actually explore new formulations.
        batches = store.list_batches(project_id)
        random_state = 42 + 1013 * int(len(batches)) + int(len(data))
    result = recommend_reformulations(
        data,
        mixture_columns=config["mixture_columns"],
        process_columns=config.get("process_columns", []),
        categorical_columns=config.get("categorical_columns", []),
        response_specs=specifications_from_config(config),
        mixture_bounds={key: tuple(value) for key, value in config["mixture_bounds"].items()},
        process_bounds={key: tuple(value) for key, value in config.get("process_bounds", {}).items()},
        category_values=config.get("category_values", {}),
        mixture_total=float(config.get("mixture_total", 100.0)),
        ingredient_to_remove=config.get("ingredient_to_remove"),
        baseline=config.get("baseline"),
        ingredient_costs=config.get("ingredient_costs", {}),
        status_column=config.get("status_column") or "status",
        n_recommendations=int(config.get("n_recommendations", 5)),
        candidate_pool_size=int(config.get("candidate_pool_size", 2500)),
        min_distance=float(config.get("min_distance", 0.06)),
        random_state=random_state,
    )
    result.model_details["random_state"] = str(random_state)
    return result


def row_meets_all_specs(row: Mapping[str, Any], specs: list[Specification]) -> bool:
    for spec in specs:
        value = row.get(spec.response)
        if value is None or pd.isna(value):
            return False
        value = float(value)
        if spec.minimum is not None and value < spec.minimum:
            return False
        if spec.maximum is not None and value > spec.maximum:
            return False
    return True


def _ratio_progress(actual: float, required: float) -> float:
    if required <= 0:
        return 1.0
    return min(max(actual / required, 0.0), 1.0)


def qualification_progress(store: ProjectStore, project_id: str) -> dict[str, Any]:
    project = store.get_project(project_id)
    config = ensure_v04_config(project["config"])
    specs = specifications_from_config(config)
    experiments = store.list_experiments(project_id, source_type="recommended")
    if experiments.empty:
        experiments = pd.DataFrame(columns=["status", "qualification_stage", "replicate_group"])
    completed = experiments[experiments.get("status", pd.Series(dtype=str)) == "completed"].copy()
    if not completed.empty:
        completed["compliant"] = [row_meets_all_specs(row, specs) for row in completed.to_dict("records")]
    else:
        completed["compliant"] = pd.Series(dtype=bool)

    replicate_data = replicate_summary(store, project_id)
    robustness_runs = store.list_robustness_runs(project_id)
    robust_probabilities: list[float] = []
    if not robustness_runs.empty:
        for value in robustness_runs["result"]:
            if isinstance(value, Mapping) and value.get("robust_success_probability") is not None:
                robust_probabilities.append(float(value["robust_success_probability"]))
    best_robust_probability = max(robust_probabilities) if robust_probabilities else None

    stage_records: list[dict[str, Any]] = []
    weighted_total = 0.0
    gates = config["qualification_gates"]
    for stage in QUALIFICATION_STAGES:
        gate = gates[stage]
        stage_completed = completed[completed.get("qualification_stage", "") == stage].copy()
        completed_count = int(len(stage_completed))
        compliant_count = int(stage_completed["compliant"].sum()) if not stage_completed.empty else 0
        success_rate = compliant_count / completed_count if completed_count else 0.0
        reasons: list[str] = []
        components: list[float] = []

        required_completed = int(gate.get("required_completed", 0))
        required_compliant = int(gate.get("required_compliant", 0))
        minimum_success_rate = float(gate.get("minimum_success_rate", 0.0))
        components.append(_ratio_progress(completed_count, required_completed))
        components.append(_ratio_progress(compliant_count, required_compliant))
        components.append(_ratio_progress(success_rate, minimum_success_rate))
        if completed_count < required_completed:
            reasons.append(f"Needs {required_completed - completed_count} more completed experiment(s)")
        if compliant_count < required_compliant:
            reasons.append(f"Needs {required_compliant - compliant_count} more compliant result(s)")
        if success_rate + 1e-12 < minimum_success_rate:
            reasons.append(f"Success rate {success_rate:.0%} is below {minimum_success_rate:.0%}")

        required_groups = int(gate.get("required_replicate_groups", 0))
        min_replicates = int(gate.get("minimum_replicates_per_group", 1))
        cv_limits = gate.get("max_cv_by_response", {}) or {}
        passing_groups = 0
        if required_groups > 0:
            stage_groups = replicate_data[
                replicate_data.get("qualification_stage", pd.Series(dtype=str)) == stage
            ] if not replicate_data.empty else pd.DataFrame()
            screened_out: list[str] = []
            for _, group in stage_groups.iterrows():
                if int(group.get("completed_replicates", 0)) < min_replicates:
                    continue
                # Stage 1 (Wheeler screen): every judged response must have
                # replicates consistent with each other before the CV means
                # anything. A group with an inconsistent replicate cannot pass.
                screen_ok = True
                for response in cv_limits:
                    consistent = group.get(f"consistent_{response}")
                    if consistent is False:
                        screen_ok = False
                        note = str(group.get(f"screen_note_{response}", "")).strip()
                        screened_out.append(
                            f"group '{group.get('replicate_group')}' ({response}: {note or 'inconsistent replicate'})"
                        )
                        break
                if not screen_ok:
                    continue
                # Stage 2: the CV check, now on replicates that agree.
                cv_ok = True
                for response, limit in cv_limits.items():
                    cv = group.get(f"cv_{response}")
                    if cv is None or pd.isna(cv) or float(cv) > float(limit):
                        cv_ok = False
                        break
                if cv_ok:
                    passing_groups += 1
            components.append(_ratio_progress(passing_groups, required_groups))
            if passing_groups < required_groups:
                reasons.append(
                    f"Needs {required_groups - passing_groups} replicate group(s) with at least {min_replicates} runs, consistent replicates, and acceptable CV"
                )
                if screened_out:
                    reasons.append(
                        "Replicate screen (per Wheeler): " + "; ".join(screened_out[:3])
                        + (" …" if len(screened_out) > 3 else "")
                    )

        min_robust = gate.get("minimum_robust_probability")
        if min_robust is not None:
            robust_value = best_robust_probability or 0.0
            components.append(_ratio_progress(robust_value, float(min_robust)))
            if robust_value + 1e-12 < float(min_robust):
                label = "none" if best_robust_probability is None else f"{best_robust_probability:.0%}"
                reasons.append(f"Best robustness result {label} is below {float(min_robust):.0%}")

        completion = float(np.mean(components)) if components else 1.0
        passed = len(reasons) == 0
        weighted_total += STAGE_WEIGHTS[stage] * completion
        stage_records.append(
            {
                "stage": stage,
                "completed": completed_count,
                "compliant": compliant_count,
                "success_rate": success_rate,
                "passing_replicate_groups": passing_groups,
                "best_robust_probability": best_robust_probability,
                "completion": completion,
                "gate_passed": passed,
                "status": "Passed" if passed else "In progress" if completed_count else "Not started",
                "remaining_requirements": "; ".join(reasons) if reasons else "All configured gates passed",
            }
        )

    stage_frame = pd.DataFrame(stage_records)
    return {
        "score": float(weighted_total),
        "all_gates_passed": bool(stage_frame["gate_passed"].all()) if not stage_frame.empty else False,
        "completed_platform_experiments": int(len(completed)),
        "compliant_platform_experiments": int(completed["compliant"].sum()) if not completed.empty else 0,
        "best_robust_probability": best_robust_probability,
        "stage_progress": stage_frame,
        "replicate_summary": replicate_data,
    }


def result_payload(result: ReformulationResult) -> dict[str, Any]:
    return {
        "decision": result.decision,
        "decision_reason": result.decision_reason,
        "warnings": list(result.warnings),
        "recommendations": result.recommendations.to_dict("records"),
        "backtest": result.backtest.to_dict("records"),
        "qualification_plan": result.qualification_plan.to_dict("records"),
        "model_details": result.model_details,
    }


def create_recommendation_batch(
    store: ProjectStore,
    project_id: str,
    *,
    trigger: str = "manual_generation",
    random_state: int | None = None,
) -> tuple[ReformulationResult, str]:
    project = store.get_project(project_id)
    upgraded = ensure_v04_config(project["config"])
    if upgraded != project["config"]:
        store.update_project_config(project_id, upgraded)
    result = run_project_model(store, project_id, random_state=random_state)
    batch_id = store.create_batch(
        project_id,
        result.recommendations,
        decision=result.decision,
        decision_reason=result.decision_reason,
    )
    progress = qualification_progress(store, project_id)
    best = result.recommendations.iloc[0]
    store.save_snapshot(
        project_id,
        batch_id=batch_id,
        trigger=trigger,
        decision=result.decision,
        best_success_probability=float(best["probability_all_specs"]),
        best_feasibility_probability=float(best["probability_feasible"]),
        qualification_score=progress["score"],
        completed_platform_experiments=progress["completed_platform_experiments"],
        compliant_platform_experiments=progress["compliant_platform_experiments"],
        result_payload=result_payload(result),
    )
    return result, batch_id


def refresh_after_result(
    store: ProjectStore,
    project_id: str,
    *,
    trigger: str = "experiment_result",
    random_state: int | None = None,
    create_next_batch_when_resolved: bool = True,
) -> tuple[ReformulationResult, str | None]:
    result = run_project_model(store, project_id, random_state=random_state)
    progress = qualification_progress(store, project_id)
    best = result.recommendations.iloc[0]
    batches = store.list_batches(project_id)
    latest_batch_id = None if batches.empty else str(batches.iloc[0]["id"])
    store.save_snapshot(
        project_id,
        batch_id=latest_batch_id,
        trigger=trigger,
        decision=result.decision,
        best_success_probability=float(best["probability_all_specs"]),
        best_feasibility_probability=float(best["probability_feasible"]),
        qualification_score=progress["score"],
        completed_platform_experiments=progress["completed_platform_experiments"],
        compliant_platform_experiments=progress["compliant_platform_experiments"],
        result_payload=result_payload(result),
    )

    new_batch_id: str | None = None
    if create_next_batch_when_resolved:
        batches = store.list_batches(project_id)
        has_open_proposal = False
        latest_resolved = False
        if not batches.empty:
            has_open_proposal = bool((batches["status"] == "proposed").any())
            latest = batches.iloc[0]
            latest_resolved = int(latest["experiment_count"] or 0) > 0 and int(latest["resolved_count"] or 0) >= int(latest["experiment_count"] or 0)
        if latest_resolved and not has_open_proposal and result.decision != "BEGIN QUALIFICATION":
            new_batch_id = store.create_batch(
                project_id,
                result.recommendations,
                decision=result.decision,
                decision_reason=result.decision_reason,
            )
    return result, new_batch_id
