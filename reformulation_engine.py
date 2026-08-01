"""Reformulation Assurance recommendation engine.

This prototype is designed for small formulation datasets. It recommends a
batch of replacement experiments while respecting mixture totals, ingredient
and process bounds, multiple product specifications, feasibility history,
ingredient cost, and distance from a proven baseline formula.

The engine deliberately combines several signals rather than presenting a
single black-box optimum:

* probability of meeting every specification;
* probability that the experiment is physically feasible;
* uncertainty and model disagreement;
* disruption from the baseline formula;
* estimated ingredient cost;
* diversity across the proposed experiment batch.

It also retrospectively compares Gaussian-process and random-forest models on
historical data so the user can see how well the modeling approach performs on
that project before relying on live recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Specification:
    """Acceptable range for one measured response."""

    response: str
    minimum: float | None = None
    maximum: float | None = None
    weight: float = 1.0

    def validate(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError(f"specification for '{self.response}' needs a minimum or maximum")
        if self.minimum is not None and not np.isfinite(self.minimum):
            raise ValueError(f"minimum for '{self.response}' must be finite")
        if self.maximum is not None and not np.isfinite(self.maximum):
            raise ValueError(f"maximum for '{self.response}' must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(f"minimum exceeds maximum for '{self.response}'")
        if self.weight <= 0:
            raise ValueError(f"weight for '{self.response}' must be positive")


@dataclass(frozen=True)
class ReformulationResult:
    recommendations: pd.DataFrame
    backtest: pd.DataFrame
    qualification_plan: pd.DataFrame
    decision: str
    decision_reason: str
    warnings: tuple[str, ...]
    model_details: dict[str, str]


class MixedFeatureEncoder:
    """Small dense encoder for numeric and categorical experiment variables."""

    def __init__(self, numeric_columns: Sequence[str], categorical_columns: Sequence[str]):
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.scaler = StandardScaler()
        self.onehot: OneHotEncoder | None = None
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> "MixedFeatureEncoder":
        if self.numeric_columns:
            self.scaler.fit(data[self.numeric_columns].astype(float))
        if self.categorical_columns:
            # sklearn renamed sparse to sparse_output in 1.2. Support both APIs.
            try:
                self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:  # pragma: no cover - older sklearn compatibility
                self.onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.onehot.fit(data[self.categorical_columns].astype(str))
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("encoder must be fitted before transform")
        parts: list[np.ndarray] = []
        if self.numeric_columns:
            parts.append(self.scaler.transform(data[self.numeric_columns].astype(float)))
        if self.categorical_columns:
            assert self.onehot is not None
            parts.append(self.onehot.transform(data[self.categorical_columns].astype(str)))
        if not parts:
            raise ValueError("at least one numeric or categorical feature is required")
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.fit(data).transform(data)


@dataclass
class _ResponseModel:
    gp: GaussianProcessRegressor
    rf: RandomForestRegressor
    encoder: MixedFeatureEncoder
    y_scale: float

    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.encoder.transform(data)
        gp_mean, gp_std = self.gp.predict(X, return_std=True)
        rf_mean = self.rf.predict(X)
        disagreement = np.abs(gp_mean - rf_mean)
        combined_std = np.sqrt(gp_std**2 + (0.5 * disagreement) ** 2)
        return gp_mean, np.maximum(combined_std, 1e-9), disagreement


# ---------------------------------------------------------------------------
# Validation and candidate generation
# ---------------------------------------------------------------------------


def _ensure_columns(data: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _clean_status(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"completed", "complete", "success", "successful", "ok", "passed"}:
        return "completed"
    if text in {"failed", "failure", "infeasible", "invalid", "separated", "unsafe"}:
        return "failed"
    return text or "unknown"


def _validate_bounds(
    columns: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    *,
    allow_fixed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    lows: list[float] = []
    highs: list[float] = []
    for column in columns:
        if column not in bounds:
            raise ValueError(f"bounds missing for '{column}'")
        lo, hi = map(float, bounds[column])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"bounds for '{column}' must be finite")
        if lo > hi or (not allow_fixed and lo == hi):
            raise ValueError(f"invalid bounds for '{column}': ({lo}, {hi})")
        lows.append(lo)
        highs.append(hi)
    return np.asarray(lows), np.asarray(highs)


def infer_numeric_bounds(
    data: pd.DataFrame,
    columns: Sequence[str],
    expansion_fraction: float = 0.05,
    floor: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Infer search limits from observed values."""
    result: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            raise ValueError(f"'{column}' has no finite values")
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo
        pad = span * expansion_fraction if span else max(abs(lo) * 0.05, 1.0)
        lower = lo - pad
        if floor is not None:
            lower = max(lower, floor)
        result[column] = (lower, hi + pad)
    return result


def sample_bounded_simplex(
    n_samples: int,
    lows: np.ndarray,
    highs: np.ndarray,
    total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample feasible mixture vectors with bounded components and fixed total.

    This uses randomized water-filling. It is not intended to be perfectly
    uniform over the polytope; it is intended to create a broad, reproducible,
    valid candidate pool for optimization.
    """
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    if lows.shape != highs.shape:
        raise ValueError("lows and highs must have matching shapes")
    if np.any(lows > highs):
        raise ValueError("all lower bounds must be <= upper bounds")
    if total < lows.sum() - 1e-9 or total > highs.sum() + 1e-9:
        raise ValueError(
            f"mixture total {total:g} is infeasible; feasible range is "
            f"[{lows.sum():g}, {highs.sum():g}]"
        )

    d = len(lows)
    output = np.empty((n_samples, d), dtype=float)
    for row in range(n_samples):
        x = lows.copy()
        remaining = float(total - x.sum())
        room = highs - x
        iterations = 0
        while remaining > 1e-9:
            active = np.flatnonzero(room > 1e-12)
            if active.size == 0:
                break
            weights = rng.dirichlet(np.ones(active.size))
            proposed = remaining * weights
            add = np.minimum(proposed, room[active])
            x[active] += add
            room[active] -= add
            remaining -= float(add.sum())
            iterations += 1
            if iterations > 10 * max(d, 1):
                # Deterministic final fill prevents rare numerical stalls.
                for idx in active:
                    delta = min(remaining, room[idx])
                    x[idx] += delta
                    room[idx] -= delta
                    remaining -= delta
                    if remaining <= 1e-9:
                        break
                break
        if abs(x.sum() - total) > 1e-6:
            raise RuntimeError("failed to generate a valid bounded mixture")
        output[row] = x
    return output


def _generate_candidates(
    *,
    mixture_columns: Sequence[str],
    process_columns: Sequence[str],
    categorical_columns: Sequence[str],
    mixture_bounds: Mapping[str, tuple[float, float]],
    process_bounds: Mapping[str, tuple[float, float]],
    category_values: Mapping[str, Sequence[object]],
    mixture_total: float,
    n_candidates: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    mix_lows, mix_highs = _validate_bounds(mixture_columns, mixture_bounds)
    mixture = sample_bounded_simplex(
        n_candidates, mix_lows, mix_highs, mixture_total, rng
    )
    result = pd.DataFrame(mixture, columns=mixture_columns)

    if process_columns:
        process_lows, process_highs = _validate_bounds(process_columns, process_bounds)
        for i, column in enumerate(process_columns):
            lo, hi = process_lows[i], process_highs[i]
            result[column] = lo if lo == hi else rng.uniform(lo, hi, size=n_candidates)

    for column in categorical_columns:
        values = list(category_values.get(column, []))
        if not values:
            raise ValueError(f"no allowed category values supplied for '{column}'")
        result[column] = rng.choice(np.asarray(values, dtype=object), size=n_candidates)
    return result


# ---------------------------------------------------------------------------
# Model fitting and retrospective evidence
# ---------------------------------------------------------------------------


def _build_gp(n_dimensions: int, random_state: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_dimensions),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-9, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        # A few extra hyperparameter-optimization restarts materially reduce
        # bad local optima on small datasets, at negligible cost at this scale.
        n_restarts_optimizer=3,
        random_state=random_state,
    )


def _fit_response_model(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    response: str,
    random_state: int,
) -> tuple[_ResponseModel, list[str]]:
    frame = data.loc[data[response].notna(), [*feature_columns, response]].copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[response] = pd.to_numeric(frame[response], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[*numeric_columns, response])
    if len(frame) < 8:
        raise ValueError(f"'{response}' needs at least 8 completed measurements")

    encoder = MixedFeatureEncoder(numeric_columns, categorical_columns)
    X = encoder.fit_transform(frame[feature_columns])
    y = frame[response].to_numpy(dtype=float)

    gp = _build_gp(X.shape[1], random_state)
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(X, y)
    if caught:
        notes.append(
            f"The Gaussian-process fit for '{response}' reached a parameter boundary."
        )

    rf = RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=max(1, min(3, len(frame) // 12)),
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    y_scale = max(float(np.nanstd(y)), 1e-9)
    return _ResponseModel(gp=gp, rf=rf, encoder=encoder, y_scale=y_scale), notes


def _backtest_response(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    response: str,
    random_state: int,
) -> list[dict[str, float | str]]:
    frame = data.loc[data[response].notna(), [*feature_columns, response]].copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[response] = pd.to_numeric(frame[response], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[*numeric_columns, response])
    n = len(frame)
    if n < 10:
        return []

    encoder = MixedFeatureEncoder(numeric_columns, categorical_columns)
    X = encoder.fit_transform(frame[feature_columns])
    y = frame[response].to_numpy(dtype=float)
    n_splits = min(5, max(3, n // 5))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # The backtest must evaluate the same model family that is deployed:
    # hyperparameters are fitted inside every CV fold, exactly as the live
    # model fits them. (Earlier versions cross-validated a GP with frozen,
    # never-fitted hyperparameters, which made the "historical model
    # evidence" table describe a model nobody was using.)
    models = {
        "Gaussian process": _build_gp(X.shape[1], random_state),
        "Random forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=max(1, min(3, n // 12)),
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    records: list[dict[str, float | str]] = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
        records.append(
            {
                "response": response,
                "model": name,
                "folds": n_splits,
                "mae": float(mean_absolute_error(y, pred)),
                "rmse": float(math.sqrt(mean_squared_error(y, pred))),
                "r2": float(r2_score(y, pred)),
            }
        )
    return records


def _fit_feasibility_model(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    status_column: str | None,
    random_state: int,
) -> tuple[RandomForestClassifier | None, MixedFeatureEncoder | None, list[str]]:
    notes: list[str] = []
    if not status_column or status_column not in data.columns:
        notes.append("No failure-status column was used; physical feasibility is assumed.")
        return None, None, notes

    frame = data.loc[:, [*feature_columns, status_column]].copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_columns)
    labels = frame[status_column].map(_clean_status)
    usable = labels.isin(["completed", "failed"])
    frame = frame.loc[usable]
    y = (labels.loc[usable] == "completed").astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    if len(frame) < 12 or np.min(counts) < 3:
        notes.append(
            "There are too few completed and failed experiments to learn feasibility reliably."
        )
        return None, None, notes

    encoder = MixedFeatureEncoder(numeric_columns, categorical_columns)
    X = encoder.fit_transform(frame[feature_columns])
    model = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, encoder, notes


# ---------------------------------------------------------------------------
# Scoring and reporting
# ---------------------------------------------------------------------------


def specification_probability(
    mean: np.ndarray,
    std: np.ndarray,
    specification: Specification,
) -> np.ndarray:
    """Probability that a normally approximated response meets its limits."""
    specification.validate()
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    lower_prob = np.zeros_like(mean)
    upper_prob = np.ones_like(mean)
    if specification.minimum is not None:
        lower_prob = norm.cdf((specification.minimum - mean) / std)
    if specification.maximum is not None:
        upper_prob = norm.cdf((specification.maximum - mean) / std)
    return np.clip(upper_prob - lower_prob, 0.0, 1.0)


def _normalized_distance(
    candidates: pd.DataFrame,
    baseline: Mapping[str, object],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    distance = np.zeros(len(candidates), dtype=float)
    dimensions = 0
    for column in numeric_columns:
        if column not in baseline:
            continue
        lo, hi = bounds[column]
        span = float(hi) - float(lo)
        # Fixed dimensions (for example the ingredient deliberately removed)
        # are not discretionary changes and must not dominate disruption.
        if span <= 1e-9:
            continue
        base = float(baseline[column])
        distance += ((candidates[column].astype(float).to_numpy() - base) / span) ** 2
        dimensions += 1
    for column in categorical_columns:
        if column not in baseline:
            continue
        distance += (candidates[column].astype(str).to_numpy() != str(baseline[column])).astype(float)
        dimensions += 1
    return np.sqrt(distance / max(dimensions, 1))


def _nearest_history_distance(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    numeric_columns: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    if not numeric_columns:
        return np.zeros(len(candidates))
    active_columns = [c for c in numeric_columns if float(bounds[c][1]) - float(bounds[c][0]) > 1e-9]
    if not active_columns:
        return np.zeros(len(candidates))
    lows = np.asarray([bounds[c][0] for c in active_columns], dtype=float)
    highs = np.asarray([bounds[c][1] for c in active_columns], dtype=float)
    spans = highs - lows
    cand = (candidates[active_columns].to_numpy(dtype=float) - lows) / spans
    hist = (history[active_columns].to_numpy(dtype=float) - lows) / spans
    return np.min(np.linalg.norm(cand[:, None, :] - hist[None, :, :], axis=2), axis=1)


def _history_extrapolation_threshold(
    history: pd.DataFrame,
    numeric_columns: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
) -> float:
    """Calibrate extrapolation distance from spacing inside historical data."""
    if len(history) < 3 or not numeric_columns:
        return 0.45
    active_columns = [c for c in numeric_columns if float(bounds[c][1]) - float(bounds[c][0]) > 1e-9]
    if len(history) < 3 or not active_columns:
        return 0.45
    lows = np.asarray([bounds[c][0] for c in active_columns], dtype=float)
    highs = np.asarray([bounds[c][1] for c in active_columns], dtype=float)
    spans = highs - lows
    points = (history[active_columns].to_numpy(dtype=float) - lows) / spans
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(max(0.40, 1.75 * np.quantile(nearest, 0.95)))


def _ingredient_cost(
    candidates: pd.DataFrame,
    mixture_columns: Sequence[str],
    ingredient_costs: Mapping[str, float] | None,
    mixture_total: float,
) -> np.ndarray:
    if not ingredient_costs:
        return np.full(len(candidates), np.nan)
    cost = np.zeros(len(candidates), dtype=float)
    for column in mixture_columns:
        unit_cost = float(ingredient_costs.get(column, 0.0))
        cost += candidates[column].to_numpy(dtype=float) / mixture_total * unit_cost
    return cost


def _normalize_benefit(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.full_like(values, 0.5, dtype=float)
    if finite.sum() < 2:
        return result
    lo, hi = np.nanpercentile(values[finite], [5, 95])
    if hi - lo < 1e-12:
        return result
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    result[finite] = scaled[finite] if higher_is_better else 1.0 - scaled[finite]
    return result


def _select_purpose_batch(
    scored: pd.DataFrame,
    n_recommendations: int,
    numeric_columns: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    min_distance: float,
) -> pd.DataFrame:
    role_metrics = [
        ("Balanced recommendation", "balanced_score"),
        ("Highest success probability", "probability_all_specs"),
        ("Lowest formulation disruption", "minimal_change_score"),
        ("Highest information value", "information_score"),
        ("Lowest projected ingredient cost", "cost_score"),
    ]
    best_success = float(scored["success_score"].max())
    viable_floor = max(0.20, 0.60 * best_success)
    chosen: list[int] = []
    purposes: list[str] = []

    def separated(index: int) -> bool:
        if not chosen or not numeric_columns:
            return True
        lows = np.asarray([bounds[c][0] for c in numeric_columns], dtype=float)
        highs = np.asarray([bounds[c][1] for c in numeric_columns], dtype=float)
        spans = np.maximum(highs - lows, 1e-9)
        point = (scored.loc[index, numeric_columns].to_numpy(dtype=float) - lows) / spans
        previous = (
            scored.loc[chosen, numeric_columns].to_numpy(dtype=float) - lows
        ) / spans
        return bool(np.all(np.linalg.norm(previous - point, axis=1) >= min_distance))

    for purpose, metric in role_metrics:
        if len(chosen) >= n_recommendations:
            break
        eligible = scored
        if purpose in {"Lowest formulation disruption", "Lowest projected ingredient cost"}:
            eligible = scored[scored["success_score"] >= viable_floor]
        elif purpose == "Highest information value":
            eligible = scored[scored["probability_feasible"] >= 0.50]
        if eligible.empty:
            eligible = scored
        order = eligible[metric].sort_values(ascending=False).index
        selected = next((int(i) for i in order if i not in chosen and separated(int(i))), None)
        if selected is None:
            selected = next((int(i) for i in order if i not in chosen), None)
        if selected is not None:
            chosen.append(selected)
            purposes.append(purpose)

    if len(chosen) < n_recommendations:
        for index in scored["balanced_score"].sort_values(ascending=False).index:
            index = int(index)
            if index in chosen:
                continue
            if separated(index) or len(chosen) + 1 == n_recommendations:
                chosen.append(index)
                purposes.append("Additional diverse candidate")
            if len(chosen) >= n_recommendations:
                break

    result = scored.loc[chosen].copy().reset_index(drop=True)
    result.insert(0, "purpose", purposes)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def _qualification_plan(best: pd.Series, response_specs: Sequence[Specification]) -> pd.DataFrame:
    spec_summary = "; ".join(
        f"{s.response}: "
        + (
            f"{s.minimum:g} to {s.maximum:g}"
            if s.minimum is not None and s.maximum is not None
            else f">= {s.minimum:g}"
            if s.minimum is not None
            else f"<= {s.maximum:g}"
        )
        for s in response_specs
    )
    return pd.DataFrame(
        [
            {
                "stage": 1,
                "qualification_step": "Confirmation replicates",
                "recommended_design": "Run the proposed formula in triplicate under nominal conditions.",
                "purpose": "Confirm repeatability and estimate laboratory noise.",
                "acceptance_gate": spec_summary,
            },
            {
                "stage": 2,
                "qualification_step": "Process-window challenge",
                "recommended_design": "Test low and high mixing temperature/time settings around nominal.",
                "purpose": "Verify the formula remains inside specification under normal process variation.",
                "acceptance_gate": "All stress points meet critical specifications.",
            },
            {
                "stage": 3,
                "qualification_step": "Raw-material variation",
                "recommended_design": "Repeat with at least three supplier lots or approved sources.",
                "purpose": "Measure sensitivity to lot and supplier variability.",
                "acceptance_gate": "No material lot causes a critical specification failure.",
            },
            {
                "stage": 4,
                "qualification_step": "Stability and aging",
                "recommended_design": "Run accelerated storage or aging tests appropriate to the product.",
                "purpose": "Detect delayed separation, cure, viscosity, appearance, or performance changes.",
                "acceptance_gate": "Aged samples remain within release and shelf-life limits.",
            },
            {
                "stage": 5,
                "qualification_step": "Pilot-scale verification",
                "recommended_design": "Produce one pilot batch using production-representative equipment.",
                "purpose": "Confirm scale-up does not change product performance.",
                "acceptance_gate": "Pilot batch meets all release specifications and processability criteria.",
            },
        ]
    )


def recommend_reformulations(
    data: pd.DataFrame,
    *,
    mixture_columns: Sequence[str],
    process_columns: Sequence[str],
    categorical_columns: Sequence[str],
    response_specs: Sequence[Specification],
    mixture_bounds: Mapping[str, tuple[float, float]],
    process_bounds: Mapping[str, tuple[float, float]],
    category_values: Mapping[str, Sequence[object]] | None = None,
    mixture_total: float = 100.0,
    ingredient_to_remove: str | None = None,
    baseline: Mapping[str, object] | None = None,
    ingredient_costs: Mapping[str, float] | None = None,
    status_column: str | None = None,
    n_recommendations: int = 5,
    candidate_pool_size: int = 7000,
    min_distance: float = 0.06,
    random_state: int = 42,
) -> ReformulationResult:
    """Generate an auditable batch of reformulation experiments."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if n_recommendations < 1:
        raise ValueError("n_recommendations must be at least 1")
    if candidate_pool_size < max(500, n_recommendations * 20):
        raise ValueError("candidate_pool_size is too small")
    if not mixture_columns:
        raise ValueError("select at least one mixture column")
    if not response_specs:
        raise ValueError("provide at least one response specification")
    for spec in response_specs:
        spec.validate()

    category_values = category_values or {}
    feature_columns = [*mixture_columns, *process_columns, *categorical_columns]
    numeric_columns = [*mixture_columns, *process_columns]
    response_columns = [spec.response for spec in response_specs]
    _ensure_columns(data, [*feature_columns, *response_columns], "required")
    if status_column:
        _ensure_columns(data, [status_column], "status")
    if ingredient_to_remove and ingredient_to_remove not in mixture_columns:
        raise ValueError("ingredient_to_remove must be a selected mixture column")

    data = data.copy()
    for column in numeric_columns + response_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    # A removed ingredient is fixed at zero in all proposed formulas.
    mixture_bounds = dict(mixture_bounds)
    if ingredient_to_remove:
        mixture_bounds[ingredient_to_remove] = (0.0, 0.0)

    # Confirm the historical data include enough usable completed rows.
    completed_mask = pd.Series(True, index=data.index)
    warnings_out: list[str] = []
    if status_column:
        statuses = data[status_column].map(_clean_status)
        completed_mask = statuses == "completed"
    completed = data.loc[completed_mask].copy()
    if len(completed) < 8:
        raise ValueError("at least 8 completed experiments are required")
    if len(completed) < 15:
        warnings_out.append(
            "Fewer than 15 completed experiments are available; treat the first batch as exploratory."
        )
    if len(completed) <= 3 * len(feature_columns):
        warnings_out.append(
            "The dataset is small relative to the number of variables; uncertainty and model disagreement matter."
        )

    candidates = _generate_candidates(
        mixture_columns=mixture_columns,
        process_columns=process_columns,
        categorical_columns=categorical_columns,
        mixture_bounds=mixture_bounds,
        process_bounds=process_bounds,
        category_values=category_values,
        mixture_total=mixture_total,
        n_candidates=candidate_pool_size,
        random_state=random_state,
    )

    all_bounds = {**mixture_bounds, **process_bounds}
    if baseline is None:
        baseline = completed.iloc[0][feature_columns].to_dict()

    model_details: dict[str, str] = {}
    backtest_records: list[dict[str, float | str]] = []
    probability_components: list[np.ndarray] = []
    weighted_components: list[np.ndarray] = []
    uncertainty_components: list[np.ndarray] = []
    disagreement_components: list[np.ndarray] = []

    for offset, spec in enumerate(response_specs):
        model, notes = _fit_response_model(
            completed,
            feature_columns,
            numeric_columns,
            categorical_columns,
            spec.response,
            random_state + offset,
        )
        warnings_out.extend(notes)
        mean, std, disagreement = model.predict(candidates)
        candidates[f"predicted_{spec.response}"] = mean
        candidates[f"uncertainty_{spec.response}"] = std
        candidates[f"probability_{spec.response}_in_spec"] = specification_probability(
            mean, std, spec
        )
        in_spec = candidates[f"probability_{spec.response}_in_spec"].to_numpy()
        probability_components.append(in_spec)
        weighted_components.append(np.power(in_spec, spec.weight))
        uncertainty_components.append(std / model.y_scale)
        disagreement_components.append(disagreement / model.y_scale)
        model_details[spec.response] = f"GP: {model.gp.kernel_}; RF: 180 trees"
        backtest_records.extend(
            _backtest_response(
                completed,
                feature_columns,
                numeric_columns,
                categorical_columns,
                spec.response,
                random_state + offset,
            )
        )

    feasibility_model, feasibility_encoder, feasibility_notes = _fit_feasibility_model(
        data,
        feature_columns,
        numeric_columns,
        categorical_columns,
        status_column,
        random_state,
    )
    warnings_out.extend(feasibility_notes)
    if feasibility_model is not None and feasibility_encoder is not None:
        feasibility_probability = feasibility_model.predict_proba(
            feasibility_encoder.transform(candidates[feature_columns])
        )[:, 1]
    else:
        feasibility_probability = np.ones(len(candidates))
    candidates["probability_feasible"] = feasibility_probability

    # Headline probability: the joint modeled probability that every response
    # meets its specification, assuming responses are independent given the
    # inputs. Specification weights and the feasibility estimate are
    # deliberately NOT blended in here — anything labeled a probability should
    # only contain probabilities.
    joint = np.ones(len(candidates))
    for component in probability_components:
        joint *= component
    candidates["probability_all_specs"] = np.clip(joint, 0.0, 1.0)

    # Ranking score: specification weights and the (uncalibrated) feasibility
    # estimate do matter when choosing which experiments to run, so they act
    # here — in a score, not in anything labeled a probability.
    weighted = np.ones(len(candidates))
    for component in weighted_components:
        weighted *= component
    weighted *= feasibility_probability
    candidates["success_score"] = np.clip(weighted, 0.0, 1.0)

    mean_uncertainty = np.mean(np.vstack(uncertainty_components), axis=0)
    mean_disagreement = np.mean(np.vstack(disagreement_components), axis=0)
    candidates["model_disagreement"] = mean_disagreement
    candidates["information_score"] = _normalize_benefit(
        mean_uncertainty * feasibility_probability, higher_is_better=True
    )
    model_details["probability_note"] = (
        "probability_all_specs multiplies per-response in-spec probabilities and assumes "
        "responses are independent given the inputs; correlations between responses are not "
        "modeled. probability_feasible is a separate random-forest estimate and is not "
        "calibrated. Candidate ranking uses success_score (specification-weighted and "
        "feasibility-adjusted), which is a score, not a probability."
    )
    model_details["information_note"] = (
        "information_score is normalized predictive uncertainty times feasibility "
        "(uncertainty sampling); it is not a formal expected-information-gain acquisition."
    )

    candidates["distance_from_baseline"] = _normalized_distance(
        candidates,
        baseline,
        numeric_columns,
        categorical_columns,
        all_bounds,
    )
    candidates["minimal_change_score"] = _normalize_benefit(
        candidates["distance_from_baseline"].to_numpy(), higher_is_better=False
    )
    candidates["nearest_history_distance"] = _nearest_history_distance(
        candidates,
        completed,
        numeric_columns,
        all_bounds,
    )
    extrapolation_threshold = _history_extrapolation_threshold(
        completed, numeric_columns, all_bounds
    )
    candidates["extrapolation_warning"] = (
        candidates["nearest_history_distance"] > extrapolation_threshold
    )
    model_details["extrapolation_threshold"] = f"Normalized distance > {extrapolation_threshold:.3f}"

    candidates["projected_ingredient_cost"] = _ingredient_cost(
        candidates, mixture_columns, ingredient_costs, mixture_total
    )
    candidates["cost_score"] = _normalize_benefit(
        candidates["projected_ingredient_cost"].to_numpy(), higher_is_better=False
    )

    # Penalize strong extrapolation and high model disagreement without hiding them.
    confidence_penalty = np.clip(1.0 - 0.35 * mean_disagreement, 0.25, 1.0)
    extrapolation_penalty = np.where(candidates["extrapolation_warning"], 0.70, 1.0)
    candidates["balanced_score"] = (
        0.52 * candidates["success_score"]
        + 0.16 * candidates["minimal_change_score"]
        + 0.14 * candidates["information_score"]
        + 0.10 * candidates["cost_score"]
        + 0.08 * _normalize_benefit(
            candidates["nearest_history_distance"].to_numpy(), higher_is_better=False
        )
    ) * confidence_penalty * extrapolation_penalty

    recommendations = _select_purpose_batch(
        candidates,
        n_recommendations,
        numeric_columns,
        all_bounds,
        min_distance,
    )

    # Put the user-facing evidence columns first.
    front = [
        "rank",
        "purpose",
        *feature_columns,
        "probability_all_specs",
        "probability_feasible",
        "projected_ingredient_cost",
        "distance_from_baseline",
        "nearest_history_distance",
        "model_disagreement",
        "extrapolation_warning",
        "success_score",
        "balanced_score",
    ]
    response_output = []
    for spec in response_specs:
        response_output.extend(
            [
                f"predicted_{spec.response}",
                f"uncertainty_{spec.response}",
                f"probability_{spec.response}_in_spec",
            ]
        )
    recommendations = recommendations[[*front, *response_output]]

    best = recommendations.iloc[0]
    best_success = float(best["probability_all_specs"])
    if best_success >= 0.80 and not bool(best["extrapolation_warning"]):
        decision = "BEGIN QUALIFICATION"
        decision_reason = (
            f"The leading candidate has a {best_success:.0%} modeled probability of meeting all "
            "specifications and is supported by nearby historical evidence."
        )
    elif best_success < 0.25:
        decision = "COLLECT FOUNDATION DATA"
        decision_reason = (
            f"The strongest candidate has only a {best_success:.0%} modeled probability of meeting all "
            "specifications. Run targeted baseline or feasibility experiments before optimization."
        )
    else:
        decision = "RUN NEXT BATCH"
        decision_reason = (
            f"The strongest candidate has a {best_success:.0%} modeled probability of meeting all "
            "specifications. The evidence supports another learning batch, but not qualification yet."
        )

    qualification = _qualification_plan(best, response_specs)
    backtest = pd.DataFrame(backtest_records)
    if backtest.empty:
        warnings_out.append(
            "Historical model backtesting requires at least 10 measured experiments per response."
        )

    return ReformulationResult(
        recommendations=recommendations,
        backtest=backtest,
        qualification_plan=qualification,
        decision=decision,
        decision_reason=decision_reason,
        warnings=tuple(dict.fromkeys(warnings_out)),
        model_details=model_details,
    )


def build_markdown_report(
    result: ReformulationResult,
    *,
    project_name: str,
    ingredient_removed: str,
) -> str:
    """Create a portable recommendation and qualification report."""
    best = result.recommendations.iloc[0]
    lines = [
        f"# Reformulation Assurance Report: {project_name}",
        "",
        f"**Ingredient removed:** {ingredient_removed}",
        f"**Decision:** {result.decision}",
        "",
        result.decision_reason,
        "",
        "## Leading recommendation",
        "",
        f"- Purpose: {best['purpose']}",
        f"- Probability of meeting all specifications: {best['probability_all_specs']:.1%}",
        f"- Probability of physical feasibility: {best['probability_feasible']:.1%}",
        f"- Distance from baseline: {best['distance_from_baseline']:.3f}",
        f"- Extrapolation warning: {'Yes' if bool(best['extrapolation_warning']) else 'No'}",
        "",
        "## Recommended experiment batch",
        "",
        result.recommendations.to_markdown(index=False),
        "",
        "## Historical model evidence",
        "",
        result.backtest.to_markdown(index=False) if not result.backtest.empty else "Not enough data for backtesting.",
        "",
        "## Path to qualification",
        "",
        result.qualification_plan.to_markdown(index=False),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- None"])
    lines.extend(
        [
            "",
            "---",
            "This prototype provides decision support. Scientists remain responsible for safety, physical feasibility, and final qualification.",
        ]
    )
    return "\n".join(lines)
