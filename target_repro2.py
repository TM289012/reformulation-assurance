"""Round 2: discover the real 0.15 target API.

Same Terminal window as before, then:  python3 target_repro2.py
Paste the full output back.
"""
import inspect
import warnings

warnings.simplefilter("ignore")

from baybe.targets import NumericalTarget

print("public attributes on NumericalTarget:")
print([m for m in dir(NumericalTarget) if not m.startswith("_")])

print("\nmatch_triangular signature:", inspect.signature(NumericalTarget.match_triangular))

print("\n--- match_triangular with cutoffs= ---")
try:
    t = NumericalTarget.match_triangular("viscosity_cp", cutoffs=(8000.0, 14000.0))
    print("worked:", t)
except Exception as exc:
    print(f"FAILED -> {type(exc).__name__}: {exc}")

candidates = ["ramp", "normalized_ramp", "normalize_ramp", "linear_ramp", "clamp",
              "clamped_affine", "normalized", "normalize", "sigmoid", "bell", "match_bell"]
for cand in candidates:
    if hasattr(NumericalTarget, cand):
        obj = getattr(NumericalTarget, cand)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(signature unavailable)"
        print(f"\nfound '{cand}' with signature {sig}")
