"""v0.4 assurance analytics: robustness, replicates, and calibration."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from project_store import ProjectStore
from reformulation_engine import (
    Specification,
    _clean_status,
    _fit_response_model,
)


def _specifications(config: Mapping[str, Any]) -> list[Specification]:
    return [Specification(**item) for item in config["response_specs"]]


def _meets_spec(value: float, spec: Specification) -> bool:
    if spec.minimum is not None and value < spec.minimum:
        return False
    if spec.maximum is not None and value > spec.maximum:
        return False
    return True


def _repair_mixture(
    row: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    total: float,
) -> np.ndarray:
    """Clip and rebalance one perturbed mixture to the bounded simplex."""
    x = np.clip(np.asarray(row, dtype=float), lows, highs)
    for _ in range(20):
        delta = float(total - x.sum())
        if abs(delta) <= 1e-8:
            return x
        room = highs - x if delta > 0 else x - lows
        active = room > 1e-12
        if not active.any():
            break
        weights = room[active] / room[active].sum()
        x[active] += np.sign(delta) * np.minimum(abs(delta) * weights, room[active])
    if abs(total - x.sum()) > 1e-6:
        raise ValueError("manufacturing variation produced an infeasible mixture")
    return x


def default_variation_config(config: Mapping[str, Any]) -> dict[str, float]:
    """Practical default 1-sigma tolerances derived from allowed ranges."""
    variation: dict[str, float] = {}
    for column, bounds in config.get("mixture_bounds", {}).items():
        lo, hi = map(float, bounds)
        variation[column] = 0.0 if lo == hi else max((hi - lo) * 0.015, 0.05)
    for column, bounds in config.get("process_bounds", {}).items():
        lo, hi = map(float, bounds)
        variation[column] = max((hi - lo) * 0.03, 0.01)
    return variation


def simulate_manufacturing_variation(
    data: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    variation_std: Mapping[str, float] | None = None,
    n_simulations: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate specification survival under manufacturing variation.

    Numeric inputs are perturbed with independent normal variation, clipped to
    project bounds, and mixture percentages are projected back to the required
    total. Response uncertainty is sampled from the fitted ensemble's predictive
    distribution, so the result includes both process variation and model noise.
    """
    if n_simulations < 50:
        raise ValueError("n_simulations must be at least 50")
    rng = np.random.default_rng(random_state)
    mixture_columns = list(config["mixture_columns"])
    process_columns = list(config.get("process_columns", []))
    categorical_columns = list(config.get("categorical_columns", []))
    numeric_columns = [*mixture_columns, *process_columns]
    feature_columns = [*numeric_columns, *categorical_columns]
    specs = _specifications(config)
    variation = {**default_variation_config(config), **(variation_std or {})}

    status_column = config.get("status_column") or "status"
    completed = data.copy()
    if status_column in completed.columns:
        completed = completed[completed[status_column].map(_clean_status) == "completed"]
    if len(completed) < 8:
        raise ValueError("at least 8 completed experiments are required for robustness simulation")

    sim = pd.DataFrame(index=np.arange(n_simulations))
    mixture_lows = np.asarray([float(config["mixture_bounds"][c][0]) for c in mixture_columns])
    mixture_highs = np.asarray([float(config["mixture_bounds"][c][1]) for c in mixture_columns])
    nominal_mix = np.asarray([float(candidate[c]) for c in mixture_columns])
    mixture_std = np.asarray([max(float(variation.get(c, 0.0)), 0.0) for c in mixture_columns])
    raw_mix = rng.normal(nominal_mix, mixture_std, size=(n_simulations, len(mixture_columns)))
    repaired = np.vstack(
        [
            _repair_mixture(row, mixture_lows, mixture_highs, float(config.get("mixture_total", 100.0)))
            for row in raw_mix
        ]
    )
    for index, column in enumerate(mixture_columns):
        sim[column] = repaired[:, index]

    for column in process_columns:
        lo, hi = map(float, config["process_bounds"][column])
        std = max(float(variation.get(column, 0.0)), 0.0)
        sim[column] = np.clip(rng.normal(float(candidate[column]), std, n_simulations), lo, hi)
    for column in categorical_columns:
        sim[column] = str(candidate[column])

    pass_matrix = np.ones((n_simulations, len(specs)), dtype=bool)
    nominal_pass_matrix = np.ones((n_simulations, len(specs)), dtype=bool)
    nominal_frame = pd.DataFrame(
        [{column: candidate[column] for column in feature_columns}] * n_simulations
    )
    response_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    for offset, spec in enumerate(specs):
        model, _ = _fit_response_model(
            completed,
            feature_columns,
            numeric_columns,
            categorical_columns,
            spec.response,
            random_state + offset,
        )
        mean, std, _ = model.predict(sim[feature_columns])
        simulated_outcome = rng.normal(mean, std)
        nominal_mean, nominal_std, _ = model.predict(nominal_frame[feature_columns])
        nominal_outcome = rng.normal(nominal_mean, nominal_std)
        passed = np.ones(n_simulations, dtype=bool)
        nominal_passed = np.ones(n_simulations, dtype=bool)
        if spec.minimum is not None:
            passed &= simulated_outcome >= spec.minimum
            nominal_passed &= nominal_outcome >= spec.minimum
        if spec.maximum is not None:
            passed &= simulated_outcome <= spec.maximum
            nominal_passed &= nominal_outcome <= spec.maximum
        pass_matrix[:, offset] = passed
        nominal_pass_matrix[:, offset] = nominal_passed
        response_records.append(
            {
                "response": spec.response,
                "mean": float(np.mean(simulated_outcome)),
                "std": float(np.std(simulated_outcome, ddof=1)),
                "p05": float(np.quantile(simulated_outcome, 0.05)),
                "p50": float(np.quantile(simulated_outcome, 0.50)),
                "p95": float(np.quantile(simulated_outcome, 0.95)),
                "probability_in_spec": float(np.mean(passed)),
            }
        )
        # Rank correlations expose which tolerances drive each response.
        for column in numeric_columns:
            if float(np.std(sim[column])) <= 1e-12:
                correlation = 0.0
            else:
                correlation = float(spearmanr(sim[column], simulated_outcome).statistic)
                if not np.isfinite(correlation):
                    correlation = 0.0
            sensitivity_records.append(
                {
                    "response": spec.response,
                    "variable": column,
                    "spearman_correlation": correlation,
                    "absolute_sensitivity": abs(correlation),
                }
            )

    all_pass = np.all(pass_matrix, axis=1)
    nominal_all_pass = np.all(nominal_pass_matrix, axis=1)
    robust_probability = float(np.mean(all_pass))
    monte_carlo_nominal_probability = float(np.mean(nominal_all_pass))
    optimizer_nominal_probability = candidate.get("probability_all_specs")
    if optimizer_nominal_probability is None and isinstance(candidate.get("recommendation"), Mapping):
        optimizer_nominal_probability = candidate["recommendation"].get("probability_all_specs")
    optimizer_nominal_probability = (
        float(optimizer_nominal_probability) if optimizer_nominal_probability is not None else None
    )
    robustness_drop = monte_carlo_nominal_probability - robust_probability

    sensitivity = pd.DataFrame(sensitivity_records).sort_values(
        ["response", "absolute_sensitivity"], ascending=[True, False]
    )
    return {
        "simulation_count": int(n_simulations),
        # Backwards-compatible key: the optimizer's point estimate.
        "nominal_success_probability": optimizer_nominal_probability,
        "optimizer_nominal_success_probability": optimizer_nominal_probability,
        # Apples-to-apples Monte Carlo estimate at exact nominal settings.
        "monte_carlo_nominal_success_probability": monte_carlo_nominal_probability,
        "robust_success_probability": robust_probability,
        "robustness_drop": robustness_drop,
        "probability_method_note": (
            "The optimizer nominal probability and Monte Carlo probabilities use different "
            "calculation methods. Compare Monte Carlo nominal with robust success to isolate "
            "the effect of manufacturing variation."
        ),
        "recommended_disposition": (
            "ROBUST ENOUGH FOR QUALIFICATION"
            if robust_probability >= 0.80
            else "TIGHTEN PROCESS WINDOW OR RUN ROBUSTNESS TESTS"
            if robust_probability >= 0.50
            else "NOT ROBUST ENOUGH"
        ),
        "response_summary": pd.DataFrame(response_records),
        "sensitivity": sensitivity,
        "variation_std": variation,
    }


def result_for_storage(result: Mapping[str, Any]) -> dict[str, Any]:
    stored = dict(result)
    for key in ("response_summary", "sensitivity"):
        if isinstance(stored.get(key), pd.DataFrame):
            stored[key] = stored[key].to_dict("records")
    return stored


def replicate_summary(
    store: ProjectStore,
    project_id: str,
) -> pd.DataFrame:
    project = store.get_project(project_id)
    responses = [item["response"] for item in project["config"]["response_specs"]]
    experiments = store.list_experiments(project_id, source_type="recommended")
    if experiments.empty:
        return pd.DataFrame()
    completed = experiments[experiments["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (stage, group), frame in completed.groupby(["qualification_stage", "replicate_group"], dropna=False):
        record: dict[str, Any] = {
            "qualification_stage": stage,
            "replicate_group": group,
            "completed_replicates": int(len(frame)),
        }
        for response in responses:
            values = pd.to_numeric(frame.get(response), errors="coerce").dropna()
            if values.empty:
                record[f"mean_{response}"] = np.nan
                record[f"cv_{response}"] = np.nan
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                record[f"mean_{response}"] = mean
                record[f"cv_{response}"] = abs(std / mean) if abs(mean) > 1e-12 else np.nan
        records.append(record)
    return pd.DataFrame(records)


def _calibration_view(
    frame: pd.DataFrame,
    specs: list[Specification],
    *,
    formulation_level: bool,
) -> dict[str, pd.DataFrame | float | None]:
    observation_records: list[dict[str, Any]] = []
    campaign_records: list[dict[str, Any]] = []

    if formulation_level:
        rows: list[dict[str, Any]] = []
        for group, grouped in frame.groupby("replicate_group", dropna=False):
            first = grouped.iloc[0]
            record = first.to_dict()
            record["experiment_code"] = str(group)
            record["replicate_count"] = int(len(grouped))
            for spec in specs:
                values = pd.to_numeric(grouped.get(spec.response), errors="coerce").dropna()
                record[spec.response] = float(values.mean()) if not values.empty else np.nan
            rows.append(record)
        evaluation = pd.DataFrame(rows)
    else:
        evaluation = frame.copy()
        evaluation["replicate_count"] = 1

    for _, row in evaluation.iterrows():
        recommendation = row.get("recommendation") or {}
        actual_all = True
        has_all = True
        for spec in specs:
            actual = row.get(spec.response)
            predicted = recommendation.get(f"predicted_{spec.response}")
            uncertainty = recommendation.get(f"uncertainty_{spec.response}")
            if actual is None or pd.isna(actual):
                has_all = False
                actual_all = False
                continue
            actual_value = float(actual)
            actual_all &= _meets_spec(actual_value, spec)
            if predicted is None or uncertainty is None:
                continue
            predicted_value = float(predicted)
            uncertainty_value = max(float(uncertainty), 1e-12)
            error = actual_value - predicted_value
            observation_records.append(
                {
                    "experiment_code": row["experiment_code"],
                    "replicate_group": row.get("replicate_group"),
                    "replicate_count": int(row.get("replicate_count", 1)),
                    "qualification_stage": row.get("qualification_stage"),
                    "response": spec.response,
                    "predicted": predicted_value,
                    "actual": actual_value,
                    "error": error,
                    "absolute_error": abs(error),
                    "standardized_error": error / uncertainty_value,
                    "inside_90_interval": abs(error) <= 1.644854 * uncertainty_value,
                }
            )
        predicted_probability = recommendation.get("probability_all_specs")
        if has_all and predicted_probability is not None:
            campaign_records.append(
                {
                    "experiment_code": row["experiment_code"],
                    "replicate_group": row.get("replicate_group"),
                    "replicate_count": int(row.get("replicate_count", 1)),
                    "predicted_probability": float(predicted_probability),
                    "actual_success": int(actual_all),
                }
            )

    observations = pd.DataFrame(observation_records)
    summaries: list[dict[str, Any]] = []
    if not observations.empty:
        for response, response_frame in observations.groupby("response"):
            errors = response_frame["error"].to_numpy(dtype=float)
            summaries.append(
                {
                    "response": response,
                    "n": int(len(response_frame)),
                    "mae": float(response_frame["absolute_error"].mean()),
                    "rmse": float(math.sqrt(np.mean(errors**2))),
                    "bias": float(np.mean(errors)),
                    "coverage_90": float(response_frame["inside_90_interval"].mean()),
                }
            )
    response_summary = pd.DataFrame(summaries)

    campaigns = pd.DataFrame(campaign_records)
    bins = pd.DataFrame()
    brier: float | None = None
    if not campaigns.empty:
        brier = float(np.mean((campaigns["predicted_probability"] - campaigns["actual_success"]) ** 2))
        campaigns["probability_bin"] = pd.cut(
            campaigns["predicted_probability"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.000001],
            include_lowest=True,
        )
        bins = (
            campaigns.groupby("probability_bin", observed=False)
            .agg(
                experiments=("actual_success", "size"),
                mean_predicted_probability=("predicted_probability", "mean"),
                observed_success_rate=("actual_success", "mean"),
            )
            .reset_index()
        )
        bins["probability_bin"] = bins["probability_bin"].astype(str)
    return {
        "observations": observations,
        "response_summary": response_summary,
        "probability_bins": bins,
        "campaigns": campaigns,
        "brier_score": brier,
    }


def calibration_report(store: ProjectStore, project_id: str) -> dict[str, Any]:
    """Return both run-level and formulation-level prospective calibration.

    Run-level evidence treats every physical replicate as a separate outcome.
    Formulation-level evidence averages linked replicates and counts the formula
    once, preventing repeated runs of one candidate from overstating dataset
    diversity.
    """
    project = store.get_project(project_id)
    specs = _specifications(project["config"])
    experiments = store.list_experiments(project_id, source_type="recommended")
    if experiments.empty:
        empty = {"observations": pd.DataFrame(), "response_summary": pd.DataFrame(), "probability_bins": pd.DataFrame(), "campaigns": pd.DataFrame(), "brier_score": None}
        return {
            **empty,
            "run_level": empty,
            "formulation_level": empty,
            "formulation_observations": pd.DataFrame(),
            "formulation_response_summary": pd.DataFrame(),
            "formulation_probability_bins": pd.DataFrame(),
            "formulation_brier_score": None,
        }
    completed = experiments[experiments["status"] == "completed"].copy()
    run_level = _calibration_view(completed, specs, formulation_level=False)
    formulation_level = _calibration_view(completed, specs, formulation_level=True)
    return {
        # Backwards-compatible run-level fields.
        **run_level,
        "run_level": run_level,
        "formulation_level": formulation_level,
        "formulation_observations": formulation_level["observations"],
        "formulation_response_summary": formulation_level["response_summary"],
        "formulation_probability_bins": formulation_level["probability_bins"],
        "formulation_brier_score": formulation_level["brier_score"],
    }
