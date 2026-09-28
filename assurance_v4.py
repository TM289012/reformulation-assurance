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


MOVING_RANGE_UPPER_FACTOR = 3.268  # D4 for n = 2: upper limit of a moving-range chart
MAX_DECIMALS = 12


def _decimal_places(value: float) -> int:
    """Decimal places a stored reading uses (5.35 gives 2, 11250.0 gives 0). The
    tolerance is a few units in the last place, so tiny readings such as 2.5e-10
    keep their decimals and large ones such as 123456789.123 are not read as whole."""
    v = float(value)
    if v == 0 or not math.isfinite(v):
        return 0
    tolerance = 4 * math.ulp(v)
    for d in range(0, MAX_DECIMALS + 1):
        if abs(round(v, d) - v) <= tolerance:
            return d
    return MAX_DECIMALS


def measurement_increment(values: list[float]) -> float:
    """The recording increment of a set of readings, read off the data (Wheeler's
    first step is to determine the measurement increment used, "by inspecting either
    the ranges or the original data"). When the lab knows its recording step it
    should say so instead: a declared step always wins (see ``recording_steps`` in
    the project configuration), because reading it off a few numbers can go wrong.

    How it is read, erring toward the coarser step (fail-safe for the chunky call and
    for :func:`cv_upper_bound`, both of which it makes stricter, though it makes the
    widened consistency screen more lenient): take the
    finest decimal place any reading uses (5.4 and 5.35 give 0.01; whole numbers give
    1), then the largest step of one or five times a power of ten of that place that
    every reading is a multiple of. Dry times logged as 35, 35, 40 give 5 and
    viscosities 11250, 11400, 12800 give 50; readings that happen to share such a
    step by chance are read coarser than they were, which the declared step fixes.

    What cannot be seen: stored numbers drop trailing zeros (pH 7.00 reads as 7.0, a
    step of 1, far coarser than 0.01); steps of 2 or 25 are read as 1 or 5; and
    readings converted between units or computed from others carry long decimals,
    so their step looks far finer than it was.
    """
    readings = [float(v) for v in values if math.isfinite(float(v))]
    if not readings:
        return 1.0
    decimals = max(_decimal_places(v) for v in readings)
    scale = 10 ** decimals
    if decimals >= MAX_DECIMALS or max(abs(v) for v in readings) * scale >= 2 ** 53:
        return 10.0 ** (-decimals)
    common = 0
    for v in readings:
        common = math.gcd(common, abs(int(round(v * scale))))
    step = 1
    power = 1
    while common and 5 * power <= common:
        if common % (5 * power) == 0:
            step = 5 * power
        if common % (10 * power) == 0:
            step = 10 * power
        power *= 10
    return step / scale


def possible_range_values(upper_range_limit: float, increment: float) -> int:
    """Possible range values within the limits of a range chart with no lower limit:
    0, one increment, two increments, ... up to the upper range limit, zero counted
    (Wheeler's Figure 3: a limit of .01810 at .001 gives 19; Figure 4: .0102 at .01
    gives 2; Table 3, n = 2: a limit of 3.69 increments gives 4)."""
    return int(upper_range_limit / increment + 1e-9) + 1 if increment > 0 else 0


def chunky_data_check(values: list[float], increment: float | None = None) -> dict[str, Any]:
    """Wheeler's chunky-data test for a moving-range chart (subgroup size 2).

    The data are chunky when the measurement increment is too large to show
    the routine variation: count the possible range values inside the range
    chart's limits (0, one increment, two increments, ... up to the upper
    range limit 3.268 x average moving range). For moving ranges, three or
    fewer possible values mean chunky data (the many zero ranges deflate the
    average range and so the limits, which then produce false alarms); four
    is the borderline-safe condition. (Wheeler's general rule for subgroups
    of three or more, four or fewer, does not apply to moving ranges.)
    Reference: D. J. Wheeler, "What is Chunky Data?", Quality Digest, Dec 2011.
    """
    ordered = [float(v) for v in values]
    increment = float(increment) if increment else measurement_increment(ordered)
    moving_ranges = [abs(ordered[j] - ordered[j - 1]) for j in range(1, len(ordered))]
    mr_bar = sum(moving_ranges) / len(moving_ranges) if moving_ranges else 0.0
    upper_range_limit = MOVING_RANGE_UPPER_FACTOR * mr_bar
    possible = possible_range_values(upper_range_limit, increment)
    return {
        "increment": increment,
        "average_moving_range": mr_bar,
        "upper_range_limit": upper_range_limit,
        "possible_values": possible,
        "chunky": possible <= 3,
        "borderline": possible == 4,
    }


def cv_upper_bound(values: list[float], increment: float) -> float:
    """The largest CV the true values behind these readings could have, if each
    reading is its true value rounded or truncated to the recording increment.

    This is the tool's own safeguard for chunky data, not a formula from Wheeler:
    it does not estimate the CV from readings the round-off has distorted, it
    bounds it. Each true value lies within one increment-wide interval of its
    reading, so by the triangle inequality the sample standard deviation of the
    true values is at most s + (increment / 2) x sqrt(n / (n - 1)), and their mean
    is at least |mean| - increment. Returns inf when the mean is within one
    increment of zero (no bound is possible) and NaN below two readings. The bound
    is only as good as the recording step it is given: exact for a correctly
    declared step, and conservative when the step read off the data errs coarse.
    """
    readings = [float(v) for v in values]
    n = len(readings)
    if n < 2:
        return math.nan
    mean = sum(readings) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in readings) / (n - 1))
    sd_max = sd + (float(increment) / 2.0) * math.sqrt(n / (n - 1))
    floor = abs(mean) - float(increment)
    return sd_max / floor if floor > 0 else math.inf


def wheeler_screen_detail(values: list[float], increment: float | None = None) -> dict[str, Any]:
    """Chunky-data check and leave-one-out consistency screen for one replicate group.

    ``increment`` is the recording step; when it is not given it is read off the
    readings with :func:`measurement_increment`.

    1. Chunky data, after Donald Wheeler ("What is Chunky Data?", 2011). His rule is
       applied to the group's own moving-range chart: readings in run order, limits
       computed in the usual way from all of them. Three or fewer possible range
       values within the limits means the recording step hides the routine
       variation. Round-off then biases estimates of spread (Wheeler shows a standard
       deviation first inflating, then collapsing toward zero, which carries over to
       a CV), so the gates judge a chunky group on :func:`cv_upper_bound` instead of
       its CV. A finer recording step is the real fix; more replicates will not
       reliably fix it.
    2. Consistency screen for small replicate sets: judge a value that looks out of
       line against XmR natural
       limits (mean +/- 2.66 x average moving range) computed from the OTHER values in
       run order. Outside, the formulation is not yet reproducible; inside, it is
       probably reproducible and the CV may quantify its repeatability. The tool tests
       every value this way, which in practice flags the odd one, because an odd
       value widens the limits of every test it is part of. Where those siblings
       are themselves chunky (two identical
       readings, say), limits built from them would be deflated by round-off and give
       false alarms, which is Wheeler's objection to chunky data, so they are widened
       by the most the round-off could hide: every true moving range is within one
       step of the recorded one, and the tested value and the centre together within
       one step. A value outside even the widened limits is inconsistent however the
       readings were rounded. The widening is this tool's own safeguard, not
       Wheeler's; indices judged that way are listed in ``widened``. It applies inside
       chunky groups too, so a wild value among identical readings is still caught.
       Siblings with four or more possible values get plain limits, which Wheeler
       calls borderline safe.

    Returns a dict with ``consistent`` (None when there are fewer than 3 readings;
    False when some value falls outside the limits built from its siblings; True
    otherwise), ``flagged`` (index of the first inconsistent value, else None),
    ``widened``, ``chunky``, ``possible_values`` (possible range values within the
    limits of the group's own moving-range chart, None below two readings) and
    ``increment`` (the recording step used).

    With 3-6 readings everything here is indicative, not definitive: the count of
    possible values and the limits are both rough at these sizes, and both depend on
    the run order of the readings.
    """
    ordered = [float(v) for v in values if math.isfinite(float(v))]
    n = len(ordered)
    step = float(increment) if increment and float(increment) > 0 else (measurement_increment(ordered) if n else 1.0)
    group = chunky_data_check(ordered, step) if n >= 2 else None
    base: dict[str, Any] = {
        "increment": step,
        "possible_values": group["possible_values"] if group else None,
        "chunky": bool(group and group["chunky"]),
    }
    if n < 3:
        return {**base, "consistent": None, "flagged": None, "widened": []}
    widened: list[int] = []
    for i in range(n):
        others = ordered[:i] + ordered[i + 1 :]
        check = chunky_data_check(others, step)
        center = sum(others) / len(others)
        mr_bar = check["average_moving_range"]
        if check["chunky"]:
            widened.append(i)
            half_width = 2.66 * (mr_bar + step) + step
        else:
            half_width = 2.66 * mr_bar
        if abs(ordered[i] - center) > half_width:
            return {**base, "consistent": False, "flagged": i, "widened": widened}
    return {**base, "consistent": True, "flagged": None, "widened": widened}


def wheeler_screen(values: list[float]) -> tuple[bool | None, int | None]:
    """``(consistent, flagged_index)`` view of :func:`wheeler_screen_detail`.
    ``consistent`` is None when there are fewer than 3 values."""
    detail = wheeler_screen_detail(values)
    return detail["consistent"], detail["flagged"]


def declared_recording_steps(config: Mapping[str, Any]) -> dict[str, float]:
    """Recording steps the project declares per response (positive numbers only)."""
    steps: dict[str, float] = {}
    for response, value in (config.get("recording_steps") or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            steps[str(response)] = number
    return steps


def replicate_summary(
    store: ProjectStore,
    project_id: str,
) -> pd.DataFrame:
    project = store.get_project(project_id)
    config = project["config"]
    responses = [item["response"] for item in config["response_specs"]]
    declared_steps = declared_recording_steps(config)
    experiments = store.list_experiments(project_id, source_type="recommended")
    if experiments.empty:
        return pd.DataFrame()
    completed = experiments[experiments["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (stage, group), frame in completed.groupby(["qualification_stage", "replicate_group"], dropna=False):
        if "replicate_index" in frame.columns:
            # Run order for the moving ranges: R1, R2, ... R10 (not the text order R1, R10, R2).
            frame = frame.sort_values("replicate_index", kind="stable")
        record: dict[str, Any] = {
            "qualification_stage": stage,
            "replicate_group": group,
            "completed_replicates": int(len(frame)),
        }
        for response in responses:
            values = pd.to_numeric(frame.get(response), errors="coerce").dropna()
            values = values[np.isfinite(values.astype(float))]
            if values.empty:
                record[f"mean_{response}"] = np.nan
                record[f"cv_{response}"] = np.nan
                record[f"cv_upper_{response}"] = np.nan
                record[f"consistent_{response}"] = None
                record[f"chunky_{response}"] = False
                record[f"screen_note_{response}"] = ""
                continue
            readings = [float(v) for v in values.tolist()]
            if "replicate_index" in frame.columns:
                labels = [int(x) if pd.notna(x) else i + 1 for i, x in enumerate(frame.loc[values.index, "replicate_index"].tolist())]
            else:
                labels = list(range(1, len(readings) + 1))
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            cv = abs(std / mean) if abs(mean) > 1e-12 else np.nan
            record[f"mean_{response}"] = mean
            record[f"cv_{response}"] = cv
            declared = declared_steps.get(response)
            detail = wheeler_screen_detail(readings, increment=declared)
            consistent, flagged, widened = detail["consistent"], detail["flagged"], detail["widened"]
            step = detail["increment"]
            source = "set for this project" if declared else "read off the readings; set it under Recording steps if that is wrong"
            record[f"consistent_{response}"] = consistent
            record[f"chunky_{response}"] = bool(detail["chunky"])
            record[f"cv_upper_{response}"] = cv_upper_bound(readings, step)
            parts: list[str] = []
            if consistent is None:
                parts.append("needs 3+ replicates to screen")
            elif consistent is False:
                index = flagged or 0
                parts.append(
                    f"replicate #{labels[index]} inconsistent with the others"
                    + (", even with limits widened for round-off" if index in widened else "")
                )
            elif widened:
                which = ", ".join(f"#{labels[index]}" for index in widened)
                parts.append(
                    f"replicates consistent ({which} judged against limits widened for round-off, because the "
                    f"readings around {'it' if len(widened) == 1 else 'them'} are too alike at this recording step)"
                )
            else:
                parts.append("replicates consistent")
            if detail["chunky"]:
                possible = int(detail["possible_values"] or 0)
                bound = record[f"cv_upper_{response}"]
                bound_text = f"{bound:.2%}" if math.isfinite(bound) else "unbounded (the mean is within one step of zero)"
                cv_text = f"{cv:.2%}" if not pd.isna(cv) else "undefined"
                whole = step >= 1 and all(float(v).is_integer() for v in readings)
                parts.append(
                    f"chunky data (Wheeler's rule): at a recording step of {step:g} only {possible} possible "
                    f"moving-range value{'' if possible == 1 else 's'} {'fits' if possible == 1 else 'fit'} within the "
                    f"limits and four are needed, so the CV of {cv_text} is not taken at face value and gates use the "
                    f"largest CV these readings allow, {bound_text}; record one more digit to measure repeatability "
                    f"properly (more replicates will not reliably fix it)"
                    + ("; counts and whole-number scores cannot take another digit" if whole else "")
                )
            if detail["chunky"] or widened:
                parts.append(f"recording step {step:g}, {source}")
            record[f"screen_note_{response}"] = ". ".join(parts)
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
