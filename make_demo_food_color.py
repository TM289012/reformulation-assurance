"""Generate ``demo_food_color_replacement.csv``, the synthetic Red No. 3 replacement history.

The scenario is a strawberry gummy colored with FD&C Red No. 3 (erythrosine). With the
color's food use ending on 15 January 2027, the brand has to move to a natural colorant and
has been screening two: a black carrot anthocyanin extract and a beet juice concentrate.
The file is the kind of history a small product-development team would have after a few
months of bench work, 90 lots in all:

* 18 legacy lots that still contain Red No. 3 (the color standard the swap is judged against),
* 72 trial lots with the two substitutes at various levels, cook temperatures and hold times,
  of which 8 failed outright (color lost during the cook, an earthy off-note, or a deposit
  that would not set).

Every number is synthetic. The response model is a plain, documented caricature of the real
chemistry, not a measurement of any product:

* Anthocyanins are redder at low pH and survive the cook better than betanin; betanin (beet)
  is bluish-red, fades with heat and time at temperature, and carries an earthy note.
* Red No. 3 is strong and clean but fades under light, so even the legacy lots do not keep
  100 percent of their color through the four-week storage test.
* Color difference is a plain CIE76 distance from the legacy color standard
  (L* 38.5, a* 46.0, b* 24.0), which is enough for a demonstration.

Run ``python make_demo_food_color.py`` to regenerate the file; the seed is fixed, so the
output is identical from run to run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "demo_food_color_replacement.csv"
SEED = 20270115  # the Red No. 3 food deadline, 15 January 2027

STANDARD_LAB = (38.5, 46.0, 24.0)
MIXTURE_TOTAL = 100.0


def _ph(citric_acid: np.ndarray) -> np.ndarray:
    # 0.8 % citric acid reads about pH 4.0, 1.6 % about pH 3.1 in this matrix.
    return 4.0 - (citric_acid - 0.8) * (0.9 / 0.8)


def _responses(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    bc = frame["ingredient_substitute_black_carrot"].to_numpy()
    bt = frame["ingredient_substitute_beet"].to_numpy()
    r3 = frame["ingredient_legacy_red3"].to_numpy()
    temp = frame["cook_temp_c"].to_numpy()
    hold = frame["hold_time_min"].to_numpy()
    ph = _ph(frame["ingredient_citric_acid"].to_numpy())
    supplier = frame["supplier_family"].to_numpy()

    # Pigment that survives the cook. Betanin degrades much faster than anthocyanins.
    excess = np.clip(temp - 110.0, 0.0, None)
    bc_eff = bc * np.exp(-0.012 * excess - 0.004 * hold)
    bt_eff = bt * np.exp(-0.045 * excess - 0.020 * hold)
    # Supplier family B's anthocyanin lot is a little weaker, C's a little stronger.
    bc_eff = bc_eff * np.where(supplier == "B", 0.92, np.where(supplier == "C", 1.05, 1.0))

    # Anthocyanin hue: red near pH 3, purple by pH 4.
    hue_factor = np.clip(1.0 - (ph - 3.1) * 0.55, 0.3, 1.0)

    a_star = 6.0 + 42.0 * bc_eff * (0.75 + 0.25 * hue_factor) + 15.0 * bt_eff + 1300.0 * r3
    b_star = 9.0 + 17.0 * bc_eff * hue_factor + 3.5 * bt_eff + 500.0 * r3
    l_star = 50.0 - 0.25 * a_star
    a_star = a_star + rng.normal(0.0, 0.7, len(frame))
    b_star = b_star + rng.normal(0.0, 0.7, len(frame))
    l_star = l_star + rng.normal(0.0, 0.5, len(frame))
    delta_e = np.sqrt(
        (l_star - STANDARD_LAB[0]) ** 2 + (a_star - STANDARD_LAB[1]) ** 2 + (b_star - STANDARD_LAB[2]) ** 2
    )

    # Colour retention after four weeks at 40 C in a light cabinet, weighted by what each
    # pigment contributes to redness.
    w_bc = 42.0 * bc_eff
    w_bt = 15.0 * bt_eff
    w_r3 = 1300.0 * r3
    total = w_bc + w_bt + w_r3 + 1e-9
    ret_bc = 93.0 - 4.0 * np.clip(ph - 3.1, 0.0, None) / 0.9
    ret_bt = 80.0 - 2.5 * np.clip(ph - 3.1, 0.0, None) / 0.9
    ret_r3 = 90.0
    retention = (w_bc * ret_bc + w_bt * ret_bt + w_r3 * ret_r3) / total
    retention = retention - 0.8 * np.clip(hold - 6.0, 0.0, None)
    retention = retention + rng.normal(0.0, 1.4, len(frame))
    retention = np.clip(retention, 40.0, 99.0)

    # Trained-panel flavour score, 1 to 10. Beet brings an earthy note.
    flavor = 8.6 - 1.4 * bt - 0.3 * bc - 0.4 * np.clip(ph - 3.7, 0.0, None) + rng.normal(0.0, 0.3, len(frame))
    flavor = np.clip(flavor, 1.0, 10.0)

    return pd.DataFrame({
        "delta_e_vs_standard": np.round(delta_e, 2),
        "a_star": np.round(a_star, 1),
        "color_retention_pct": np.round(retention, 1),
        "flavor_score": np.round(flavor, 1),
        "_ph": ph,
        "_bt_eff": bt_eff,
    })


def _base_mixture(n: int, rng: np.random.Generator) -> pd.DataFrame:
    glucose = rng.uniform(39.0, 45.0, n)
    sugar = rng.uniform(29.0, 35.0, n)
    gelatin = rng.uniform(6.8, 8.4, n)
    citric = rng.uniform(0.9, 1.6, n)
    return pd.DataFrame({
        "ingredient_glucose_syrup": glucose,
        "ingredient_sugar": sugar,
        "ingredient_gelatin": gelatin,
        "ingredient_citric_acid": citric,
    })


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    legacy = _base_mixture(18, rng)
    legacy["ingredient_legacy_red3"] = rng.uniform(0.030, 0.032, 18)
    legacy["ingredient_substitute_black_carrot"] = 0.0
    legacy["ingredient_substitute_beet"] = 0.0
    legacy["cook_temp_c"] = rng.uniform(112.0, 122.0, 18)
    legacy["hold_time_min"] = rng.uniform(3.0, 9.0, 18)
    legacy["supplier_family"] = rng.choice(["A", "B", "C"], 18, p=[0.5, 0.3, 0.2])

    trials = _base_mixture(72, rng)
    trials["ingredient_legacy_red3"] = 0.0
    # Three screening campaigns: beet-led, black-carrot-led, and blends.
    campaign = rng.choice(["beet", "carrot", "blend"], 72, p=[0.3, 0.4, 0.3])
    bc = np.where(
        campaign == "beet", rng.uniform(0.0, 0.3, 72),
        np.where(campaign == "carrot", rng.uniform(0.4, 1.3, 72), rng.uniform(0.3, 0.9, 72)),
    )
    bt = np.where(
        campaign == "beet", rng.uniform(0.8, 2.1, 72),
        np.where(campaign == "carrot", rng.uniform(0.0, 0.3, 72), rng.uniform(0.3, 1.2, 72)),
    )
    trials["ingredient_substitute_black_carrot"] = bc
    trials["ingredient_substitute_beet"] = bt
    trials["cook_temp_c"] = rng.uniform(108.0, 126.0, 72)
    trials["hold_time_min"] = rng.uniform(2.0, 12.0, 72)
    trials["supplier_family"] = rng.choice(["A", "B", "C"], 72, p=[0.45, 0.35, 0.2])

    frame = pd.concat([legacy, trials], ignore_index=True)
    order = rng.permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)

    pigments = (
        frame["ingredient_legacy_red3"]
        + frame["ingredient_substitute_black_carrot"]
        + frame["ingredient_substitute_beet"]
    )
    fixed = (
        frame["ingredient_glucose_syrup"] + frame["ingredient_sugar"]
        + frame["ingredient_gelatin"] + frame["ingredient_citric_acid"] + pigments
    )
    frame["ingredient_water"] = MIXTURE_TOTAL - fixed

    responses = _responses(frame, rng)
    frame = pd.concat([frame, responses], axis=1)

    # Failures. The tool treats failed lots as evidence too, so a few belong in the history.
    status = np.full(len(frame), "completed", dtype=object)
    reason = np.full(len(frame), "", dtype=object)
    cook_loss = (frame["ingredient_substitute_beet"] > 1.2) & (frame["cook_temp_c"] > 122.0)
    earthy = frame["flavor_score"] < 5.8
    set_fail = (frame["ingredient_gelatin"] < 7.0) & (frame["hold_time_min"] > 10.0)
    for mask, text in ((cook_loss, "color lost during cook"), (earthy, "earthy off-note"), (set_fail, "deposit did not set")):
        idx = np.where(mask & (status == "completed"))[0]
        status[idx] = "failed"
        reason[idx] = text
    # Failed lots have no usable measurements beyond the failure itself.
    failed = status == "failed"
    for column in ("delta_e_vs_standard", "a_star", "color_retention_pct", "flavor_score"):
        frame.loc[failed, column] = np.nan
    frame["status"] = status
    frame["failure_reason"] = reason

    frame.insert(0, "experiment_id", [f"LOT-{i + 1:04d}" for i in range(len(frame))])
    columns = [
        "experiment_id",
        "ingredient_glucose_syrup", "ingredient_sugar", "ingredient_water", "ingredient_gelatin",
        "ingredient_citric_acid", "ingredient_legacy_red3", "ingredient_substitute_black_carrot",
        "ingredient_substitute_beet", "cook_temp_c", "hold_time_min", "supplier_family",
        "delta_e_vs_standard", "a_star", "color_retention_pct", "flavor_score", "status", "failure_reason",
    ]
    frame = frame[columns]
    rounding = {
        "ingredient_glucose_syrup": 4, "ingredient_sugar": 4, "ingredient_water": 4, "ingredient_gelatin": 4,
        "ingredient_citric_acid": 4, "ingredient_legacy_red3": 4, "ingredient_substitute_black_carrot": 4,
        "ingredient_substitute_beet": 4, "cook_temp_c": 2, "hold_time_min": 2,
    }
    frame = frame.round(rounding)
    # Keep the mixture summing to exactly 100 after rounding by absorbing the residue in water.
    mixture = [c for c in frame.columns if c.startswith("ingredient_")]
    frame["ingredient_water"] = (frame["ingredient_water"] + (MIXTURE_TOTAL - frame[mixture].sum(axis=1))).round(4)
    return frame


def main() -> None:
    frame = build()
    frame.to_csv(OUTPUT, index=False)
    completed = frame[frame["status"] == "completed"]
    legacy = completed[completed["ingredient_legacy_red3"] > 0]
    candidates = completed[completed["ingredient_legacy_red3"] == 0]
    passing = candidates[
        (candidates["delta_e_vs_standard"] <= 3.0) & (candidates["a_star"] >= 38.0)
        & (candidates["color_retention_pct"] >= 85.0) & (candidates["flavor_score"] >= 7.5)
    ]
    print(f"wrote {OUTPUT.name}: {len(frame)} lots, {int((frame['status'] == 'failed').sum())} failed")
    print(f"legacy lots: {len(legacy)}, candidate lots: {len(candidates)}, candidates meeting every spec: {len(passing)}")


if __name__ == "__main__":
    main()
