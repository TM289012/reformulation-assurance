"""Predictive ablation of the boundary-aware feature representations.

The sequential benchmark (backtest_ranker_vs_baybe.py) asks how many rounds a
ranker needs; on the shipped cosmetics history the control already finds a
top-2% candidate in one or two rounds, so that metric has little room to show
a difference between encodings. This script asks the question the encodings
actually bear on: does the GP+RF response model predict held-out lots better,
and with better-calibrated uncertainty, when ingredients that can be absent
are encoded with presence indicators (and the composition in ilr coordinates)
instead of raw mass fractions? Same models, same hyperparameter search, same
rows; only the encoding differs (see composition_features.py).

Two conditions:

  cv        Repeated 5-fold cross-validation over the completed lots of the
            cosmetics history, models refit inside every fold. Metrics per
            response: RMSE relative to the response's standard deviation, mean
            negative log predictive density under the model's normal
            predictive distribution (lower is better; it rewards both accuracy
            and honest uncertainty), and coverage of the nominal 90% interval
            (0.90 is calibrated). The held-out rows are also split into the
            lots that resemble the reformulation target (legacy emulsifier
            absent) and the rest.
  transfer  Train on the lots that still contain the legacy emulsifier,
            predict the lots without it: the extrapolation a reformulation
            actually asks for. Here a presence indicator for the legacy
            ingredient is constant in the training lots and is dropped, so
            "presence" can only differ from "raw" through the glucoside
            indicator; the ilr arm differs most, because the removed
            ingredient sits at its replacement value in every target lot.

Usage:
  python ablation_boundary_features.py                 # both conditions
  python ablation_boundary_features.py --repeats 1     # quicker
Results: backtest_ablation_predictive.csv (one row per response, encoding, fold)
and a printed markdown summary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_ranker_vs_baybe import feature_lists, load_cosmetics  # noqa: E402
from composition_features import REPRESENTATIONS, composition_spec_from_config  # noqa: E402
from reformulation_engine import _fit_response_model  # noqa: E402

Z90 = float(norm.ppf(0.95))


def fit_and_predict(train, test, config, response, representation, random_state=42):
    numeric, categorical, features = feature_lists(config)
    composition = None
    if representation != "raw":
        composition = composition_spec_from_config(config, representation)
        features = list(dict.fromkeys([*numeric, *composition.mixture_columns, *categorical]))
    model, notes = _fit_response_model(
        train, features, numeric, categorical, response, random_state=random_state, composition=composition
    )
    mean, std, _ = model.predict(test[features])
    return mean, std, notes


def score(y, mean, std) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    std = np.maximum(std, 1e-9)
    z = (y - mean) / std
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "nlpd": float(np.mean(0.5 * z**2 + np.log(std) + 0.5 * np.log(2 * np.pi))),
        "coverage90": float(np.mean(np.abs(z) <= Z90)),
        "n": int(len(y)),
    }


def run_cv(completed, config, representations, repeats, verbose):
    responses = [s["response"] for s in config["response_specs"]]
    removed = config["ingredient_to_remove"]
    splitter = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=2026)
    records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(completed)):
        train, test = completed.iloc[train_idx], completed.iloc[test_idx]
        target_like = test[removed].to_numpy(dtype=float) <= 0
        for response in responses:
            y = test[response].to_numpy(dtype=float)
            for representation in representations:
                started = time.time()
                mean, std, _ = fit_and_predict(train, test, config, response, representation)
                for subset, mask in (("all", np.ones(len(y), bool)), ("legacy_free", target_like), ("legacy", ~target_like)):
                    if mask.sum() == 0:
                        continue
                    records.append(
                        {
                            "condition": "cv",
                            "fold": fold,
                            "response": response,
                            "representation": representation,
                            "subset": subset,
                            **score(y[mask], mean[mask], std[mask]),
                        }
                    )
                if verbose:
                    print(f"  cv fold {fold:>2} {response:<16} {representation:<13} {time.time() - started:4.1f}s", flush=True)
    return records


def run_transfer(completed, config, representations, verbose):
    responses = [s["response"] for s in config["response_specs"]]
    removed = config["ingredient_to_remove"]
    legacy_free = completed[removed].to_numpy(dtype=float) <= 0
    train, test = completed[~legacy_free], completed[legacy_free]
    records = []
    for response in responses:
        y = test[response].to_numpy(dtype=float)
        for representation in representations:
            started = time.time()
            mean, std, notes = fit_and_predict(train, test, config, response, representation)
            records.append(
                {
                    "condition": "transfer",
                    "fold": 0,
                    "response": response,
                    "representation": representation,
                    "subset": "legacy_free",
                    **score(y, mean, std),
                }
            )
            if verbose:
                print(f"  transfer {response:<16} {representation:<13} {time.time() - started:4.1f}s  {' '.join(n for n in notes if n.startswith('Features'))}", flush=True)
    return records


def summarize(results: pd.DataFrame, completed: pd.DataFrame, config: dict) -> None:
    responses = [s["response"] for s in config["response_specs"]]
    scale = {r: float(completed[r].std(ddof=1)) for r in responses}

    def table(frame: pd.DataFrame, title: str) -> None:
        print(f"\n{title}\n")
        print("| response | encoding | RMSE / sd | NLPD (standardized) | 90% coverage | held-out rows |")
        print("|----------|----------|-----------|---------------------|--------------|---------------|")
        for response in responses:
            block = frame[frame["response"] == response]
            for representation in dict.fromkeys(block["representation"]):
                rows = block[block["representation"] == representation]
                weights = rows["n"].to_numpy(dtype=float)
                rmse = float(np.sqrt(np.average(rows["rmse"] ** 2, weights=weights)))
                # NLPD of the response divided by its standard deviation, so the
                # four responses read on one scale; differences between encodings
                # within a response are unchanged by this shift.
                nlpd = float(np.average(rows["nlpd"], weights=weights)) - np.log(scale[response])
                cover = float(np.average(rows["coverage90"], weights=weights))
                print(
                    f"| {response} | {representation} | {rmse / scale[response]:.3f} | "
                    f"{nlpd:.3f} | {cover:.2f} | {int(weights.sum())} |"
                )

    cv = results[results["condition"] == "cv"]
    if len(cv):
        table(cv[cv["subset"] == "all"], "### Cross-validation, all held-out lots")
        table(cv[cv["subset"] == "legacy_free"], "### Cross-validation, held-out lots without the legacy emulsifier")
    transfer = results[results["condition"] == "transfer"]
    if len(transfer):
        table(transfer, "### Regime transfer: trained on legacy lots, predicting the legacy-free lots")
    print(
        "\nRMSE / sd: root-mean-square error divided by the response's standard deviation over"
        " the completed lots (lower is better). NLPD: mean negative log predictive density of"
        " the standardized response under N(mean, std) (lower is better; it rewards accuracy"
        " and honest uncertainty together). Coverage: share of held-out lots inside the"
        " nominal 90% interval (0.90 is calibrated; above is over-wide, below over-confident)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="predictive ablation of boundary-aware encodings")
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--repeats", type=int, default=3, help="repeats of 5-fold cross-validation")
    parser.add_argument("--conditions", default="cv,transfer")
    parser.add_argument("--output", default="backtest_ablation_predictive.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    representations = [r.strip() for r in args.representations.split(",") if r.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    data, config = load_cosmetics()
    completed = data[data["status"].astype(str).str.lower().eq("completed")].reset_index(drop=True)
    print(f"{len(completed)} completed lots, {int((completed[config['ingredient_to_remove']] <= 0).sum())} without the legacy emulsifier", flush=True)

    records = []
    if "cv" in conditions:
        records.extend(run_cv(completed, config, representations, args.repeats, not args.quiet))
    if "transfer" in conditions:
        records.extend(run_transfer(completed, config, representations, not args.quiet))
    results = pd.DataFrame(records)
    output_path = ROOT / args.output
    results.to_csv(output_path, index=False)
    summarize(results, completed, config)
    print(f"\nPer-fold rows: {output_path.name}")


if __name__ == "__main__":
    main()
