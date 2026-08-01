# v0.6.2 Architecture

## Scientific layer

- `reformulation_engine.py`: candidate generation and response modeling
- `assurance_v4.py`: robustness, repeatability, and two-level calibration
- `process_window.py`: bounded qualification-study design
- `closed_loop.py`: retraining and qualification gates

## Evidence layer

- `project_store.py`: projects, immutable recommendation batches, experiments, replicates, batch closure, snapshots, robustness, and audit events
- `dossier.py`: evidence hashing and qualification-package export

## Product layer

- `product_store.py`: users, organizations, roles, approvals, and dossiers
- `pilot_store.py`: invitations, reset links, comments, assignments, multi-signer policies, artifact and backup metadata
- `app.py`: Streamlit interface

## v0.6.2 control changes

- Result validation is enforced in the persistence layer, not only in the UI.
- Replicate creation uses a target-total operation, making repeated clicks idempotent.
- Process-window designs carry a SHA-256 design fingerprint to prevent exact duplicate studies.
- Batch closure is explicit and auditable.
- Calibration evidence is stored and exported at both physical-run and formulation levels.
- Robustness separates optimizer scoring from same-method Monte Carlo nominal and varied-process estimates.
