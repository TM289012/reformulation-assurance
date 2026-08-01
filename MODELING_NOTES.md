# Modeling notes — what the numbers are and are not

This project prefers honest labels over impressive ones. This page says
exactly what the statistical machinery does, in plain terms, including the
parts a professional statistician would want disclosed.

## Response models

Each response (e.g. adhesion, viscosity) gets two models fitted on the
completed experiments:

- A **Gaussian process** (Matérn 5/2 kernel with per-variable length scales,
  white-noise term, normalized targets, several hyperparameter-optimization
  restarts). This provides the mean prediction and a predictive standard
  deviation.
- A **random forest** used as a second opinion. The reported uncertainty
  inflates the GP standard deviation with GP-vs-forest disagreement:
  `sqrt(gp_std² + (0.5·|gp − rf|)²)`. This is a pragmatic ensemble heuristic,
  not a calibrated posterior.

Minimum data: 8 completed measurements per response, with explicit warnings
below 15. A GP with per-variable length scales on so few points in many
dimensions is genuinely underdetermined — the warnings are part of the
product, not decoration.

## Candidate generation is not Bayesian optimization

Candidates come from random sampling of the bounded mixture simplex plus
process ranges, then get scored by the models and ranked. There is no
acquisition-function optimization (no expected improvement, no BoTorch).
This is honest random search with model-based ranking. The candidate pool
seed is derived from the project state, so the same state reproduces the
same candidates (auditable), while every new batch or recorded result
refreshes the pool (the loop keeps exploring).

## Probability semantics

- `probability_<response>_in_spec`: normal-CDF probability that one response
  lands inside its specification window, given the model's mean and
  uncertainty.
- `probability_all_specs`: the product of those per-response probabilities.
  **This assumes responses are independent given the inputs.** Correlations
  between responses (common in formulations) are not modeled, so treat this
  as an optimistic-leaning estimate.
- `probability_feasible`: a separate random-forest classifier estimate that
  the experiment can be physically completed, learned from failed/infeasible
  history. It is **not calibrated** and is never blended into
  `probability_all_specs`.
- `success_score` and `balanced_score`: ranking scores. Specification weights
  and feasibility act here. Scores are for choosing experiments; they are not
  probabilities and are never labeled as such.
- `information_score`: normalized predictive uncertainty × feasibility —
  uncertainty sampling, not a formal expected-information-gain acquisition.

## Backtests evaluate the deployed model family

The "historical model evidence" table cross-validates the same GP
construction the platform deploys, with hyperparameters fitted inside every
fold. (Earlier versions cross-validated a GP with frozen hyperparameters,
which described a model nobody was using; that is fixed.)

## Calibration is the point

At recommendation time predictions are frozen. When physical results arrive,
the platform scores those frozen predictions: error metrics, 90%-interval
coverage, and Brier scores, at both the physical-run level and the
replicate-aggregated formulation level. Caveat: at pilot scale the sample
sizes are small, so calibration evidence should be read as a trend, not a
verdict.

## Qualification gates are bookkeeping, not statistics

Stage gates count completed and compliant runs, replicate variability, and
robustness thresholds. They make evidence auditable and hard to skip. They
are not hypothesis tests and carry no confidence statements.

## Known limitations, openly

Small-data GPs in many dimensions; independence assumption in the joint
probability; uncalibrated feasibility classifier; random search rather than
acquisition-optimized proposals; Monte Carlo robustness limited by model
quality; no units/test-method schema yet (planned). If you are a formulator
or statistician and see something worse than what's listed here, please open
an issue — that is exactly the feedback this project wants.
