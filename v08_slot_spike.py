"""v0.8 spike: can BayBE's slot-based representation answer our real question?

The question our cosmetics demo actually asks is "WHICH emulsifier should
replace the legacy PEG, and at what percent?" Today the app expresses that as
separate ingredient columns. Slot-based flips it: one emulsifier SLOT with a
label parameter (which substance, encoded from real molecular structure) and
an amount parameter (how much).

This is a SPIKE on the v08-slot-based branch: standalone, not integrated with
the app, measurements are FAKE (hand-invented to echo the demo's intuition),
and success means exactly one thing - the slot machinery runs end to end on
real chemistry and returns recommendations. Requires the chemistry extras:

    pip install "baybe[chem]==0.15.*"

Substances: SMILES pulled from PubChem by hand (CIDs noted below), except the
legacy PEG. PEG-100 stearate is a polydisperse polymer with no discrete
structure on PubChem, so we use a hand-written representative oligomer
(stearate head, 8 glycol units) and say so. Every SMILES, including that one,
is validated with rdkit before anything else runs.

Run:  python v08_slot_spike.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The emulsifier menu: real molecules, hand-collected from PubChem.
# ---------------------------------------------------------------------------
SUBSTANCES = {
    # PubChem CID 62142
    "decyl_glucoside": "CCCCCCCCCCO[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O",
    # PubChem CID 10893439
    "lauryl_glucoside": "CCCCCCCCCCCCOC1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O",
    # PubChem CID 14871 (monolaurin)
    "glyceryl_laurate": "CCCCCCCCCCCC(=O)OCC(CO)O",
    # PubChem CID 24699 (monostearin)
    "glyceryl_stearate": "CCCCCCCCCCCCCCCCCC(=O)OCC(CO)O",
    # HAND-WRITTEN representative: PEG-100 stearate is polydisperse (no
    # discrete PubChem structure), so this is a stearate head with an 8-unit
    # PEG tail standing in for the real ~100-unit goo. Representative only.
    "legacy_PEG_stearate_rep": "CCCCCCCCCCCCCCCCCC(=O)OCCOCCOCCOCCOCCOCCOCCOCCO",
}


def step(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def validate_smiles() -> None:
    """Every structure gets machine-checked before use, especially the
    hand-written one. Hand-written chemistry does not get a free pass here."""
    from rdkit import Chem

    for name, smiles in SUBSTANCES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"INVALID SMILES for {name}: {smiles}")
        step(f"SMILES ok: {name} ({mol.GetNumAtoms()} heavy atoms)")


def build_searchspace_and_objective():
    from baybe.objectives import SingleTargetObjective
    from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter
    from baybe.searchspace import SearchSpace
    from baybe.targets import NumericalTarget

    try:
        emulsifier = SubstanceParameter(
            name="emulsifier", data=SUBSTANCES, encoding="MORDRED"
        )
        step("SubstanceParameter built with MORDRED encoding")
    except Exception as exc:
        step(f"MORDRED encoding failed ({type(exc).__name__}: {exc}); trying default")
        emulsifier = SubstanceParameter(name="emulsifier", data=SUBSTANCES)
        step("SubstanceParameter built with default encoding")

    amount = NumericalDiscreteParameter(
        name="emulsifier_pct", values=np.arange(1.0, 5.01, 0.5).round(1).tolist()
    )
    temperature = NumericalDiscreteParameter(
        name="emulsification_temp_c", values=[70.0, 75.0, 80.0]
    )
    searchspace = SearchSpace.from_product([emulsifier, amount, temperature])
    objective = SingleTargetObjective(target=NumericalTarget(name="stability_score"))
    step(f"searchspace built: {len(SUBSTANCES)} substances x 9 amounts x 3 temps")
    return searchspace, objective


def fake_measurements() -> pd.DataFrame:
    """FAKE data, invented for the spike, loosely echoing the demo's story:
    glucosides promising, legacy PEG solid, glycerol esters middling."""
    rows = [
        ("decyl_glucoside", 3.0, 75.0, 9.1),
        ("decyl_glucoside", 2.0, 70.0, 8.4),
        ("lauryl_glucoside", 3.0, 75.0, 8.8),
        ("lauryl_glucoside", 4.0, 80.0, 8.2),
        ("glyceryl_laurate", 3.0, 75.0, 7.6),
        ("glyceryl_stearate", 3.5, 80.0, 8.0),
        ("legacy_PEG_stearate_rep", 3.5, 75.0, 8.9),
        ("legacy_PEG_stearate_rep", 2.0, 70.0, 8.1),
    ]
    return pd.DataFrame(
        rows,
        columns=["emulsifier", "emulsifier_pct", "emulsification_temp_c", "stability_score"],
    )


def main() -> None:
    step("validating SMILES with rdkit")
    validate_smiles()

    step("building searchspace and objective")
    searchspace, objective = build_searchspace_and_objective()

    measurements = fake_measurements()
    step(f"using {len(measurements)} FAKE measurements (spike only)")

    from baybe.recommenders import BotorchRecommender

    step("asking BotorchRecommender for 3 slot recommendations")
    recommendation = BotorchRecommender().recommend(
        3, searchspace, objective, measurements
    )
    print("\n=== recommended next experiments (slot view) ===")
    print(recommendation.to_string(index=False))
    print(
        "\nSpike verdict: slot-based representation runs end to end on real "
        "chemistry. The model chose WHICH substance and HOW MUCH in one move."
    )


if __name__ == "__main__":
    main()
