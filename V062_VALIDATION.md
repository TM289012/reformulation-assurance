# Reformulation Assurance v0.6.2 Validation

## Scope

v0.6.2 is a usability and qualification-workflow release. It adds a process-window designer and closes the specific control gaps found during the first hands-on pilot rehearsal.

## Automated results

Run:

```bash
python -m unittest discover -s tests -v
```

Result: **25/25 tests passed**.

The six new v0.6.2 tests verify:

1. A process-window matrix preserves the confirmed formulation, varies bounded process controls, and creates experiments in the `process_window` stage.
2. Completed experiments cannot be saved with missing or zero response measurements.
3. A target replicate count creates only missing runs and blocks duplicate creation at the same target.
4. Batches can be formally closed while unresolved runs are cancelled and retained in the audit history.
5. Calibration separates physical-run evidence from replicate-aggregated formulation evidence.
6. Duplicate unscoped signatures for the same signer, stage, and evidence hash are rejected.
7. Robustness results expose both the optimizer nominal estimate and an apples-to-apples Monte Carlo nominal estimate.

All 19 controls from v0.4 through v0.6 continue to pass.

## End-to-end regression

`python demo_v06.py` completed successfully after the upgrade and produced:

- accepted invitation
- project comments and assignments
- completed multi-role approval policy
- encrypted dossier round trip
- verified backup
- PostgreSQL migration bundle
- 18 dossier files, including separate run-level and formulation-level calibration exports

## Browser validation limit

The Streamlit package was not installed in the build environment. The interface was syntax-compiled, and all underlying workflows were validated through automated and command-line tests. The local browser should still be tested after installation using the included demo data.
