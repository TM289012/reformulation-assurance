"""Semi-synthetic benchmark: our GP+RF ranking vs BayBE's recommender vs random.

The question: starting from the same real experiment history, which selection
brain finds a qualifying reformulation (all specs passed, legacy emulsifier
absent) in fewer new experiments?

Why not replay the dataset's own rows as candidates? Because the cosmetics
demo history contains only 13 swap-eligible lots, 5 of which qualify - a pool
that small produces ceiling effects where every method looks the same. So the
benchmark instead works the way the app actually works: candidates are fresh
formulations from the app's own bounded-simplex generator (legacy ingredient
locked to zero), and lab results for those new candidates come from a
semi-synthetic oracle.

Protocol, per repetition seed:
  1. Oracle: one random forest per response, fit on the 81 completed real
     lots. Its noiseless prediction is treated as ground truth; measurement
     noise equal to its training residual scale is added to what the arms see.
  2. Candidate pool: 400 formulations from the app's generator (same bounds
     as the demo project, legacy = 0). Identical pool for all arms.
  3. All arms start with the full real history (88 lots) as known
     measurements. Each round the arm picks one pool candidate; the oracle's
     noisy measurement of it is revealed. Success = the round in which the
     arm first picks a candidate whose *noiseless* oracle responses pass all
     specs. Censored at --max-rounds.

Arms:
  random        Uniform choice. The null hypothesis.
  ours          The app's joint in-spec probability ranking: same GP
                construction the app deploys plus RF-disagreement uncertainty
                inflation, turned into P(all specs pass). Mirrors the
                "Highest success probability" purpose. Pure exploitation.
  baybe_scalar  BayBE 0.15 stateless BotorchRecommender with the v1
                scalarized single target ("spec_margin" - worst normalized
                in-spec margin across responses, failed real lots = -1).
                Kept as the handicapped comparison point.
  baybe_desir   BayBE at full strength: native multi-target
                DesirabilityObjective, one normalized target per response
                with the project's spec weights - the same information
                structure the "ours" arm enjoys. Target constructors try the
                modern 0.15 API first, then the deprecated-but-working ones.
  Both BayBE arms need `pip install "baybe==0.15.*"`; skipped politely if
  absent. Two oracle families (rf, mlp) run by default so oracle-model bias
  is controlled rather than merely disclosed.

Disclosed limitations of v1: the oracle is itself a model (random forest, so
half of the "ours" ensemble family - a bias we accept and state); real
failure modes (broken emulsions) are not simulated for new candidates; one
dataset; 5 repetitions. Columns constant across the candidate pool (the
removed legacy ingredient, locked at 0) are excluded from both informed
arms' features: BayBE rejects constant parameters, and dropping them for
"ours" as well keeps the information sets identical.

Usage:
  python backtest_ranker_vs_baybe.py --smoke            # wiring check
  python backtest_ranker_vs_baybe.py                    # full run
  python backtest_ranker_vs_baybe.py --arms ours,random # skip baybe
Results: backtest_results.csv + a printed markdown table for the report.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from demo_seed import cosmetics_config  # noqa: E402
from reformulation_engine import (  # noqa: E402
    MixedFeatureEncoder,
    Specification,
    _fit_response_model,
    _generate_candidates,
    specification_probability,
)

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
POOL_SIZE = 400


# ---------------------------------------------------------------------------
# Data, oracle, and helpers
# ---------------------------------------------------------------------------

def load_cosmetics() -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(ROOT / "demo_cosmetics_emulsifier_swap.csv")
    config = cosmetics_config(data)
    numeric = [*config["mixture_columns"], *config["process_columns"]]
    for column in [*numeric, *[s["response"] for s in config["response_specs"]]]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data, config


def feature_lists(config: dict) -> tuple[list, list, list]:
    numeric = [*config["mixture_columns"], *config["process_columns"]]
    categorical = list(config["categorical_columns"])
    return numeric, categorical, [*numeric, *categorical]


class Oracle:
    """Per-response ground-truth models fit on the completed real lots.

    Two families are supported so results can be checked for oracle-family
    bias: "rf" (random forest - shares a family with half the app's
    ensemble) and "mlp" (a standardized neural network - structurally
    unlike both the app's GP and its RF).
    """

    def __init__(self, data: pd.DataFrame, config: dict, kind: str = "rf", random_state: int = 7):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        numeric, categorical, features = feature_lists(config)
        self.kind = kind
        self.features = features
        self.responses = [s["response"] for s in config["response_specs"]]
        self.models: dict[str, tuple[MixedFeatureEncoder, object, float]] = {}
        completed = data[data["status"].astype(str).str.lower().eq("completed")]
        for response in self.responses:
            frame = completed.dropna(subset=[response])
            encoder = MixedFeatureEncoder(numeric, categorical)
            X = encoder.fit_transform(frame[features])
            y = frame[response].to_numpy(dtype=float)
            if kind == "rf":
                model = RandomForestRegressor(
                    n_estimators=400, min_samples_leaf=2, random_state=random_state, n_jobs=-1
                )
            elif kind == "mlp":
                model = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        max_iter=6000,
                        random_state=random_state,
                    ),
                )
            else:
                raise ValueError(f"unknown oracle kind '{kind}'")
            model.fit(X, y)
            residual_std = float(np.std(y - model.predict(X)))
            noise = max(residual_std, 1e-6)
            self.models[response] = (encoder, model, noise)

    def true_responses(self, candidates: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=candidates.index)
        for response, (encoder, model, _) in self.models.items():
            out[response] = model.predict(encoder.transform(candidates[self.features]))
        return out

    def measure(self, candidates: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = self.true_responses(candidates)
        for response, (_, _, noise) in self.models.items():
            out[response] = out[response] + rng.normal(0.0, noise, size=len(out))
        return out


def passes_specs(responses: pd.DataFrame, config: dict) -> pd.Series:
    ok = pd.Series(True, index=responses.index)
    for spec in config["response_specs"]:
        values = responses[spec["response"]]
        if spec.get("minimum") is not None:
            ok &= values >= spec["minimum"]
        if spec.get("maximum") is not None:
            ok &= values <= spec["maximum"]
    return ok


def spec_margin(frame: pd.DataFrame, config: dict, reference: pd.DataFrame) -> pd.Series:
    """Worst normalized in-spec margin; rows with failed status or NaN -> -1."""
    margins = pd.Series(np.inf, index=frame.index, dtype=float)
    for spec in config["response_specs"]:
        values = frame[spec["response"]]
        completed = reference[spec["response"]].dropna()
        scale = float(completed.quantile(0.75) - completed.quantile(0.25)) or 1.0
        parts = []
        if spec.get("minimum") is not None:
            parts.append((values - spec["minimum"]) / scale)
        if spec.get("maximum") is not None:
            parts.append((spec["maximum"] - values) / scale)
        if parts:
            margins = pd.concat([margins, pd.concat(parts, axis=1).min(axis=1)], axis=1).min(axis=1)
    if "status" in frame:
        margins[frame["status"].astype(str).str.lower() != "completed"] = -1.0
    return margins.fillna(-1.0).clip(lower=-1.0)


def generate_pool(config: dict, seed: int, size: int) -> pd.DataFrame:
    return _generate_candidates(
        mixture_columns=config["mixture_columns"],
        process_columns=config["process_columns"],
        categorical_columns=config["categorical_columns"],
        mixture_bounds={k: tuple(v) for k, v in config["mixture_bounds"].items()},
        process_bounds={k: tuple(v) for k, v in config["process_bounds"].items()},
        category_values=config["category_values"],
        mixture_total=float(config["mixture_total"]),
        n_candidates=size,
        random_state=1000 + seed,
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def pick_random(rng, measurements, remaining, config) -> int:
    return int(rng.integers(len(remaining)))


def pick_ours(rng, measurements, remaining, config) -> int:
    numeric, categorical, features = feature_lists(config)
    active = config.get("_active_columns")
    if active is not None:
        numeric = [c for c in numeric if c in active]
        categorical = [c for c in categorical if c in active]
        features = [*numeric, *categorical]
    joint = np.ones(len(remaining))
    for spec_dict in config["response_specs"]:
        spec = Specification(
            response=spec_dict["response"],
            minimum=spec_dict.get("minimum"),
            maximum=spec_dict.get("maximum"),
            weight=spec_dict.get("weight", 1.0),
        )
        model, _ = _fit_response_model(
            measurements, features, numeric, categorical, spec.response, random_state=42
        )
        mean, std, _ = model.predict(remaining[features])
        joint *= specification_probability(mean, std, spec)
    return int(np.argmax(joint))


def _build_target(name: str, lo, hi, values: pd.Series):
    """Construct one normalized BayBE target, tolerant of 0.15 API variants.

    Tries the modern constructors first, then the legacy ones that the
    deprecation policy keeps alive. Returns (target, how) where `how` records
    which constructor worked, for honest reporting.
    """
    from baybe.targets import NumericalTarget

    errors: list[str] = []
    if lo is not None and hi is not None:
        bounds = (float(lo), float(hi))
        try:
            return NumericalTarget.match_triangular(name, bounds), "match_triangular"
        except Exception as exc:
            errors.append(f"match_triangular: {type(exc).__name__}: {exc}")
        try:
            return (
                NumericalTarget(name=name, mode="MATCH", bounds=bounds, transformation="TRIANGULAR"),
                "legacy MATCH",
            )
        except Exception as exc:
            errors.append(f"legacy MATCH: {type(exc).__name__}: {exc}")
    else:
        descending = lo is None  # only an upper limit -> smaller is better
        cut = float(hi if lo is None else lo)
        extreme = float(values.min() if descending else values.max())
        bounds = (min(cut, extreme), max(cut, extreme))
        if bounds[0] == bounds[1]:
            bounds = (bounds[0] - 1.0, bounds[1] + 1.0)
        try:
            return (
                NumericalTarget.normalize_ramp(name, cutoffs=bounds, descending=descending),
                "normalize_ramp",
            )
        except Exception as exc:
            errors.append(f"normalize_ramp: {type(exc).__name__}: {exc}")
        try:
            mode = "MIN" if descending else "MAX"
            return (
                NumericalTarget(name=name, mode=mode, bounds=bounds, transformation="LINEAR"),
                f"legacy {mode}",
            )
        except Exception as exc:
            errors.append(f"legacy MIN/MAX: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not construct target '{name}': " + " | ".join(errors))


def _build_desirability(config: dict, data: pd.DataFrame):
    """Multi-target desirability objective (BayBE's native multi-spec mode)."""
    from baybe.objectives import DesirabilityObjective

    targets, weights, notes = [], [], []
    for spec in config["response_specs"]:
        name = spec["response"]
        target, how = _build_target(
            name, spec.get("minimum"), spec.get("maximum"), data[name].dropna()
        )
        targets.append(target)
        weights.append(float(spec.get("weight", 1.0)))
        notes.append(f"{name}={how}")
    try:
        objective = DesirabilityObjective(targets=targets, weights=weights)
    except Exception:
        objective = DesirabilityObjective(targets=targets)
        notes.append("weights unsupported; equal weights used")
    return objective, notes


class BaybeArm:
    """BayBE arm with the v1 scalarized single target ("spec_margin")."""

    label = "baybe_scalar"

    def __init__(self, pool: pd.DataFrame, history: pd.DataFrame, config: dict):
        from baybe.objectives import SingleTargetObjective
        from baybe.recommenders import BotorchRecommender
        from baybe.searchspace import SearchSpace, SubspaceDiscrete
        from baybe.targets import NumericalTarget

        self._searchspace_cls = SearchSpace
        self._subspace_cls = SubspaceDiscrete
        self.config = config
        self.numeric, self.categorical, self.params = feature_lists(config)
        self.objective = SingleTargetObjective(target=NumericalTarget(name="spec_margin"))
        self.recommender = BotorchRecommender()
        self.reference = history

    def _measurement_table(self, measurements: pd.DataFrame, config: dict, active: list) -> pd.DataFrame:
        table = measurements[active].copy()
        table["spec_margin"] = spec_margin(measurements, config, self.reference).to_numpy()
        return table

    def pick(self, rng, measurements, remaining, config) -> int:
        # Stateless mode has no already-measured bookkeeping (that lives in
        # Campaigns), so the search space is rebuilt each round from only the
        # still-unpicked candidates - by construction every recommendation is
        # a legal pick. Measurements (real history + prior picks) sit outside
        # the space and act purely as surrogate training data. Columns that
        # are constant across the remaining candidates (e.g. the removed
        # legacy ingredient, always 0) are excluded: BayBE rejects constant
        # parameters, and for fairness the "ours" arm drops them too.
        active = [c for c in self.params if remaining[c].nunique() > 1]
        if not active:
            return 0  # one candidate effectively left; no decision to make
        subspace = self._subspace_cls.from_dataframe(
            remaining[active].reset_index(drop=True)
        )
        searchspace = self._searchspace_cls(discrete=subspace)
        table = self._measurement_table(measurements, config, active)
        recommendation = self.recommender.recommend(
            1, searchspace, self.objective, table
        )
        row = recommendation.iloc[0]
        mask = np.ones(len(remaining), dtype=bool)
        for column in [c for c in self.numeric if c in active]:
            mask &= np.isclose(remaining[column].to_numpy(dtype=float), float(row[column]), atol=1e-6)
        for column in [c for c in self.categorical if c in active]:
            mask &= remaining[column].astype(str).to_numpy() == str(row[column])
        matches = np.flatnonzero(mask)
        if len(matches) == 0:
            raise RuntimeError(
                "BayBE recommended a point not in the remaining pool "
                "(likely an already-measured row; check re-candidacy settings)."
            )
        return int(matches[0])


class BaybeDesirabilityArm(BaybeArm):
    """BayBE arm at full strength: native multi-target desirability objective.

    One normalized target per response, weighted per the project config, so
    BayBE models each response separately - the same information structure
    the "ours" arm enjoys. Failed real lots (NaN responses) are dropped from
    this arm's measurements, since desirability targets need numeric values;
    the scalarized arm instead encodes them as margin -1. Disclosed asymmetry.
    """

    label = "baybe_desir"

    def __init__(self, pool: pd.DataFrame, history: pd.DataFrame, config: dict):
        super().__init__(pool, history, config)
        self.objective, notes = _build_desirability(config, history)
        self.responses = [s["response"] for s in config["response_specs"]]
        print(f"    (baybe_desir targets: {', '.join(notes)})", flush=True)

    def _measurement_table(self, measurements: pd.DataFrame, config: dict, active: list) -> pd.DataFrame:
        table = measurements[[*active, *self.responses]].copy()
        return table.dropna(subset=self.responses)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(arm_name, pick, data, config, oracle, seed, pool_size, max_rounds, verbose=True):
    rng = np.random.default_rng(10_000 + seed)
    pool = generate_pool(config, seed, pool_size)
    true_responses = oracle.true_responses(pool)
    truth_pass = passes_specs(true_responses, config)
    true_margins = spec_margin(true_responses, config, data)
    top_threshold = float(np.quantile(true_margins, 0.98))
    base_rate = float(truth_pass.mean())

    # Information parity: columns constant across the candidate pool carry no
    # decision signal and BayBE cannot model them, so every informed arm sees
    # the same reduced feature set.
    _, _, params = feature_lists(config)
    config = {**config, "_active_columns": [c for c in params if pool[c].nunique() > 1]}

    measurements = data.copy()
    remaining = pool.copy()
    remaining_pass = truth_pass.copy()
    remaining_margin = true_margins.copy()
    rounds_to_pass = None
    for round_number in range(1, max_rounds + 1):
        start = time.time()
        position = pick(rng, measurements, remaining, config)
        chosen = remaining.iloc[[position]]
        chosen_passes = bool(remaining_pass.iloc[position])
        chosen_top = bool(remaining_margin.iloc[position] >= top_threshold)
        noisy = oracle.measure(chosen, rng)
        new_row = chosen.copy()
        for response in noisy.columns:
            new_row[response] = noisy[response].to_numpy()
        new_row["status"] = "completed"
        measurements = pd.concat([measurements, new_row], ignore_index=True)
        remaining = remaining.drop(remaining.index[position]).reset_index(drop=True)
        remaining_pass = remaining_pass.drop(remaining_pass.index[position]).reset_index(drop=True)
        remaining_margin = remaining_margin.drop(remaining_margin.index[position]).reset_index(drop=True)
        if chosen_passes and rounds_to_pass is None:
            rounds_to_pass = round_number
        if verbose:
            print(
                f"    [{arm_name} seed={seed}] round {round_number:>2}: "
                f"passes={chosen_passes} top2%={chosen_top} ({time.time() - start:.1f}s)",
                flush=True,
            )
        if chosen_top:
            return round_number, rounds_to_pass, base_rate
    return None, rounds_to_pass, base_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="ours vs baybe vs random, semi-synthetic replay")
    parser.add_argument("--arms", default="random,ours,baybe_scalar,baybe_desir")
    parser.add_argument("--oracles", default="rf,mlp")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    max_rounds, pool_size = args.max_rounds, args.pool_size
    oracle_kinds = [o.strip() for o in args.oracles.split(",") if o.strip()]
    if args.smoke:
        seeds, max_rounds, pool_size = seeds[:1], 4, 120
        oracle_kinds = oracle_kinds[:1]

    data, config = load_cosmetics()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    records = []
    for oracle_kind in oracle_kinds:
        print(f"=== oracle: {oracle_kind} ===", flush=True)
        oracle = Oracle(data, config, kind=oracle_kind)
        for seed in seeds:
            pool = generate_pool(config, seed, pool_size)
            history = data
            arms: dict[str, object] = {}
            for name in arm_names:
                if name == "random":
                    arms[name] = pick_random
                elif name == "ours":
                    arms[name] = pick_ours
                elif name in ("baybe", "baybe_scalar", "baybe_desir"):
                    arm_cls = BaybeDesirabilityArm if name == "baybe_desir" else BaybeArm
                    try:
                        arms[name] = arm_cls(pool, history, config).pick
                    except ImportError as exc:
                        print(f"SKIPPING {name} arm (not installed): {exc}")
                    except Exception as exc:
                        print(f"SKIPPING {name} arm (setup failed): {type(exc).__name__}: {exc}")
                else:
                    raise SystemExit(f"unknown arm '{name}'")

            for arm_name, pick in arms.items():
                print(f"  running oracle={oracle_kind} arm={arm_name} seed={seed} ...", flush=True)
                rounds_top, rounds_pass, base_rate = replay(
                    arm_name, pick, data, config, oracle, seed, pool_size, max_rounds
                )
                records.append(
                    {
                        "oracle": oracle_kind,
                        "arm": arm_name,
                        "seed": seed,
                        "rounds_to_top2pct": rounds_top,
                        "rounds_to_first_pass": rounds_pass,
                        "pool_base_rate": base_rate,
                    }
                )
                outcome = rounds_top if rounds_top is not None else f"censored@{max_rounds}"
                print(
                    f"  -> {oracle_kind}/{arm_name} seed={seed}: top2% in {outcome}, "
                    f"first pass {rounds_pass} (pool pass rate {base_rate:.1%})",
                    flush=True,
                )

    results = pd.DataFrame(records)
    results.to_csv(ROOT / "backtest_results.csv", index=False)

    print("\n### Experiments needed to find a top-2% formulation (lower is better)\n")
    print(f"Cosmetics demo project. Candidate pool: {pool_size} fresh formulations from the")
    print("app's generator (legacy emulsifier = 0), identical for all arms. All arms start")
    print(f"with the full real history. {len(seeds)} repetitions per oracle, censored at")
    print(f"{max_rounds} rounds. Target: a pool candidate in the top 2% by true worst-spec")
    print("margin. Two oracle families to control for oracle-model bias.\n")

    def summary_table(frame: pd.DataFrame, column: str) -> None:
        print("| oracle | arm | median rounds | min - max | censored runs |")
        print("|--------|-----|---------------|-----------|---------------|")
        for oracle_kind in dict.fromkeys(frame["oracle"]):
            block = frame[frame["oracle"] == oracle_kind]
            for arm_name in dict.fromkeys(block["arm"]):
                subset = block[block["arm"] == arm_name][column]
                finished = subset.dropna()
                censored = int(subset.isna().sum())
                if len(finished):
                    print(
                        f"| {oracle_kind} | {arm_name} | {finished.median():.0f} | "
                        f"{int(finished.min())} - {int(finished.max())} | {censored} |"
                    )
                else:
                    print(f"| {oracle_kind} | {arm_name} | - | - | {censored} |")

    summary_table(results, "rounds_to_top2pct")
    rate = results["pool_base_rate"].mean()
    print(f"\nSecondary metric, rounds to the first merely-passing candidate")
    print(f"(easy at a {rate:.0%} mean pool pass rate, shown for context):\n")
    summary_table(results, "rounds_to_first_pass")
    print("\nRaw per-run rows: backtest_results.csv")


if __name__ == "__main__":
    main()
