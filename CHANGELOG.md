# Changelog

## v0.6.3

- Security: the admin email outbox is now organization-scoped (`PilotStore.list_outbox`). Administrators can no longer see other organizations' invitation links or notifications.
- Security: password-reset links are never rendered in the interface. An administrator who could read another user's reset link could take over that account and sign approvals as them. Reset delivery now requires SMTP, or the server-side `reset_password_cli.py` helper.
- Folded in the v0.6.2.1 hotfix (`pd.Interval` serialization in dossier generation).
- Added cross-organization outbox isolation regression tests (suite is now 29 tests).
- Statistics honesty pass: `probability_all_specs` is now a pure joint probability (independence assumption stated) with specification weights and the uncalibrated feasibility estimate moved to a clearly-named `success_score`; backtests now cross-validate the deployed GP with per-fold hyperparameter fitting instead of a frozen-hyperparameter stand-in; GP fits use multiple optimizer restarts; the candidate-pool seed is derived from project state so the closed loop actually explores new candidates each round while staying reproducible. See `MODELING_NOTES.md`.
- Removed the stale `sample_assurance_report.md` (it showed pre-v0.6.3 probability semantics); regenerate samples with `python demo_v04.py`.

## v0.6.2

- Added a dedicated Process Window Designer.
- Added one-factor-at-a-time, corner-plus-center, and full three-level grid designs.
- Added process-window batch creation with qualification stage set automatically.
- Added completed-result validation for blank, nonfinite, and zero measurements.
- Added explicit persistent save confirmations.
- Added replicate-group progress and idempotent target replicate counts.
- Blocked accidental duplicate unscoped signatures.
- Displayed invitation/reset links in the email outbox.
- Split calibration into run-level and formulation-level evidence.
- Added Monte Carlo nominal probability for a direct robustness comparison.
- Added formal batch closure and bulk cancellation.
- Updated dossier exports and validation tests.

## v0.6

- Added invitations, password resets, comments, assignments, multi-signer policies, encrypted artifacts, verified backups, and PostgreSQL migration tooling.
