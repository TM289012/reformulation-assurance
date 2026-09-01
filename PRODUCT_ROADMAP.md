# Product Roadmap

The project's goal at this stage is domain signal and honest evidence, not sales: get the workflow in front of real formulators and statisticians, collect criticism, and let the calibration pages prove or disprove the models in public.

## Shipped: v0.10.0 — The Wheeler release (September 2026)

The confirmation gate is now two-stage, following the procedure Donald J. Wheeler (2010 Deming Medalist) described in correspondence: look at the replicate running record first, judge any suspect value against limits computed from the other replicates, and only let the CV mean anything once the replicates agree. The drift checker also stopped issuing hollow all-clears below the small-sample floors of an XmR chart (first/last point can't flag below 6 values, middle points below 8, limits stabilize ~17+). See CHANGELOG.md.

## Shipped: v0.9.0 — Excel round-trip (August 2026)

The workbench now hands the analysis back to spreadsheet-land: a one-click Excel workbook export with every evidence table as a tab and the evidence hash on the cover sheet. Import their workbook, analyze, and return everything as the file format formulators actually live in. Prompted by practitioner feedback that this belongs as a feature of the tools people already use, not a separate destination.

## Shipped: v0.8.0 — Signed evidence snapshots (August 2026)

Sign-offs now store the full frozen evidence snapshot (canonical JSON) alongside the SHA-256 hash and the signature, so an auditor can always see the exact evidence a signature covered instead of only proving that it changed. Adopted from the practice recommended by the eLabFTW maintainer: sign a snapshot, store both together, never bind signatures to live data. See CHANGELOG.md.

## Shipped: v0.7.0 — Frictionless entry (August 2026)

The app is clickable in a browser with nothing to install (public demo sandbox), two standalone single-question tools serve the questions formulators hit mid-experiment (baseline drift, replicate noise), and the recommendation layer was benchmarked head-to-head against BayBE 0.15 with the protocol, biases, and results public. See CHANGELOG.md and the v0.7.0 release.

## Next: v0.11 — Chemistry-aware suggestions (in design on branch `v08-slot-based`, named before renumbering)

- **Slot-based ingredient representation.** Instead of one column per candidate emulsifier, model the substitution question the way it is actually asked: one emulsifier slot choosing WHICH substance (BayBE `SubstanceParameter`, encoded from real molecular structures) and HOW MUCH, hybrid with traditional columns for everything else. Day-one spike is complete and on the branch (`v08_slot_spike.py`): five real molecules with PubChem-sourced SMILES (the polymeric legacy PEG as a disclosed representative oligomer), MORDRED descriptors, stateless recommender end to end.
- **Optional BayBE ranking mode in the app.** The benchmark showed BayBE's recommender with a proper multi-target objective performs on par with the built-in ranking; offering it as an optional mode (own candidate pool via `SearchSpace.from_dataframe`, stateless `recommend`) is now justified. Behind an optional extras install (`baybe` pulls PyTorch); the core app stays light.
- **Sparse-history benchmark.** v2 of the benchmark started every arm with 88 informative measurements, friendly territory for pure exploitation. The v3 question: how do the arms compare when history is thin, where Bayesian optimization's exploration should earn its keep?
- **In-app drift detection.** Apply the standalone drift checker's rules (XmR limits, run rules) to each project's own prospective-calibration residuals, so a shifting instrument or process flags itself.
- **SMILES support in the import wizard.** An optional molecular-structure column, so slot-based modeling and chemical encodings become available to real datasets, not just the demo.

## Later, if real users ask

- Stability-study scheduling and timepoint tracking
- Supplier-lot study designer improvements and process-window result visualization
- PostgreSQL as the active runtime, managed backups, email delivery
- Dataset/version rollback and admin-visible diagnostics

## Explicitly parked (enterprise-era items)

SSO/MFA, fine-grained permissions, deployment automation, penetration testing, billing, SLAs, and ELN/LIMS connectors are parked until there is evidence of demand from real teams. They belong to a sales motion this project is not currently running.

Regulated electronic-signature or quality-system claims remain a separate program and are not implied by any version above.
