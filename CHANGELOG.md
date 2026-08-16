# Changelog

## v0.8.0 (2026-08-08)

Signatures now preserve the exact evidence they covered. Previously a sign-off stored only the SHA-256 hash of the evidence bundle: enough to prove the evidence changed after signing, but not enough to show an auditor what was actually signed. Each signature now stores the full frozen canonical-JSON snapshot alongside the hash, adopting the practice recommended by Nicolas CARPi, maintainer of eLabFTW: sign a snapshot, store the snapshot with the signature, and never bind signatures to live data.

- Approvals now store an `evidence_snapshot` column, the canonical JSON frozen at signing. Existing databases are migrated automatically; signatures made before this release remain valid and are labeled as hash-only.
- Approvals page: a per-signature viewer shows whether the stored snapshot still re-hashes to the hash recorded at signing, and offers the snapshot as a JSON download.
- Dossier ZIP: a `signed_evidence_snapshots/` folder carries each signature's frozen evidence, and the HTML approval table gains a `snapshot_stored` column. `approvals.csv` stays readable; the snapshot text is excluded there.
- Stale-signature flagging is unchanged, but its meaning is cleaner: it now reads as "the newest evidence has not been signed yet" rather than "an old signature broke."
- Fixed stale v0.6.2 version stamps in generated dossiers.

## v0.7.0 (2026-08-05)

The frictionless-entry release: the app is now clickable in a browser with nothing to install, two standalone single-question tools serve the questions formulators hit mid-experiment, and the recommendation layer was benchmarked head-to-head against BayBE with the protocol and results public.

- Added a public demo mode (`REFORMULATION_DEMO_MODE`) for hosted sandboxes: seeds a shared demo workspace with the cosmetics emulsifier-swap project preloaded, one-click demo sign-in, and sandbox warning banners. Off by default; regular deployments are unchanged.
- Added `docs/drift-checker.html`: a standalone, dependency-free, single-page baseline drift checker (XmR individuals chart with limit and run rules). Paste time-ordered measurements of anything that should be stable; runs entirely in the browser.
- Added `docs/replicate-checker.html`: companion single-page tool answering "is the difference between two formulas real, or replicate noise?" — Welch-style standard-error screen with conservative small-sample thresholds, minimum detectable difference, and a replicates-needed estimate.
- README: added the live browser demo link and a Single-question tools section.
- Added `backtest_ranker_vs_baybe.py`: semi-synthetic replay benchmark comparing the app's joint-probability ranking, BayBE 0.15's stateless BotorchRecommender, and random selection on the cosmetics demo project. Protocol biases and v1 limitations are disclosed in the module docstring.

## v0.6.3

- Security: the admin email outbox is now organization-scoped (`PilotStore.list_outbox`). Administrators can no longer see other organizations' invitation links or notifications.
- Security: password-reset links are never rendered in the interface. An administrator who could read another user's reset link could take over that account and sign approvals as them. Reset delivery now requires SMTP, or the server-side `reset_password_cli.py` helper.
- Folded in the v0.6.2.1 hotfix (`pd.Interval` serialization in dossier generation).
- Added cross-organization outbox isolation regression tests (suite is now 29 tests).
- Statistics honesty pass: `probability_all_specs` is now a pure joint probability (independence assumption stated) with specification weights and the uncalibrated feasibility estimate moved to a clearly-named `success_score`; backtests now cross-validate the deployed GP with per-fold hyperparameter fitting instead of a frozen-hyperparameter stand-in; GP fits use multiple optimizer restarts; the candidate-pool seed is derived from project state so the closed loop actually explores new candidates each round while staying reproducible. See `MODELING_NOTES.md`.
- Removed the stale `sample_assurance_report.md` (it showed pre-v0.6.3 probability semantics); regenerate samples with `python demo_v04.py`.
- Added a second demo dataset, `demo_cosmetics_emulsifier_swap.csv`: replacing a legacy PEG emulsifier in an oil-in-water lotion (88 historical lots including 7 failed emulsions).

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
