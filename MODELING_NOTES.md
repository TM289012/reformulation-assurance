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

## Boundary-aware encodings: an ablation, not a swap (v0.12.9)

Two reviewers on the Materials Science Community Discourse (November 2026)
made the same point from different directions. In a reformulation the hard
part of the input space is the boundary, where an ingredient goes from
present to absent, and a stationary kernel on raw mass fractions treats
"absent" as merely "a very low dose". One suggested factoring each slot into
a presence indicator plus the active dosage; the other suggested testing
that, with isometric log-ratio coordinates for the composition, as an
ablation against the current kernel rather than a replacement, so the
benchmark stays interpretable. `composition_features.py` is that ablation.
The app still fits on raw fractions and nothing changes for a user; the
encodings exist so they can be measured, and the measurements are below.

**The encodings.** Both are derived from the raw columns alone: the same
rows and the same information, differently represented.

- `presence`: a 0/1 indicator for every ingredient that is absent from some
  fitted lots and present in others (in the cosmetics history, the legacy
  emulsifier and the glucoside co-emulsifier), with the raw fraction kept as
  the dosage. The GP's per-variable length scales can then put a short scale
  on the jump and a long one on the dose.
- `presence_ilr`: those indicators plus the isometric log-ratio coordinates
  of the whole composition in place of its raw fractions. Zeros are handled
  by multiplicative replacement (Martín-Fernández et al., 2003): each zero
  becomes 65 percent of the smallest non-zero amount of that ingredient in
  the fitted lots and the rest of the row shrinks so the total is unchanged.
  The ilr basis is a sequential binary partition that follows the problem:
  the reformulation slot (removed ingredient and its substitutes) against the
  rest of the formula, then the removed ingredient against its substitutes,
  then each group in column order. The test suite checks the basis is an
  isometry of the Aitchison geometry.

**Two measurements.** First, the sequential benchmark
(`backtest_ranker_vs_baybe.py --arms ours,ours_presence,ours_presence_ilr,random`),
with the same oracles, candidate pools and seeds as the BayBE comparison; the
control is the shipped ranking arm, byte for byte. It was run from the full
88-lot history and from a 30-lot subsample drawn per seed (`--history-size 30`),
since representation should matter more when data are scarce. Second, a
predictive check (`ablation_boundary_features.py`): three repeats of 5-fold
cross-validation over the 81 completed lots, GP hyperparameters refit in
every fold, scored by RMSE relative to the response's standard deviation,
the negative log predictive density of the standardized response (accuracy
and honest uncertainty together), and 90 percent interval coverage; plus a
regime-transfer condition, trained on the lots that still contain the legacy
emulsifier and predicting the lots without it, which is the extrapolation a
reformulation actually asks for.

Rounds to a top-2% candidate, median (min to max) over 5 seeds, 400-candidate
pool, censored at 40:

| history | oracle | ours (control) | ours_presence | ours_presence_ilr | random |
|---------|--------|----------------|---------------|-------------------|--------|
| 88 lots | rf  | 2 (1 to 2) | 2 (1 to 2) | 1 (1 to 4)  | 30 (7 to 38), 1 censored |
| 88 lots | mlp | 4 (1 to 5) | 4 (1 to 7) | 3 (1 to 16) | 4 of 5 censored |
| 30 lots | rf  | 2 (1 to 8) | 4 (1 to 5) | 2 (1 to 3)  | 30 (7 to 38), 1 censored |
| 30 lots | mlp | 2 (1 to 19) | 7 (2 to 10) | 4 (1 to 15) | 4 of 5 censored |

Cross-validation over all held-out lots (RMSE / sd, then NLPD, then coverage):

| response | raw | presence | presence_ilr |
|----------|-----|----------|--------------|
| viscosity_cp | 0.306, 0.32, 0.88 | 0.347, 0.54, 0.83 | 0.310, 0.30, 0.90 |
| ph | 1.042, 1.60, 0.78 | 1.052, 1.68, 0.77 | 1.101, 1.73, 0.75 |
| stability_score | 0.742, 1.30, 0.79 | 0.687, 1.21, 0.81 | 0.906, 1.43, 0.78 |
| spreadability | 0.733, 1.41, 0.77 | 0.742, 1.45, 0.76 | 0.622, 1.14, 0.86 |

In the regime-transfer condition the three encodings are within 0.1 RMSE / sd
of one another on every response (`ablation_boundary_features.py` prints the
table), and the presence indicator for the legacy ingredient is dropped
because it is constant in the training lots.

**What this says.** On this history, no encoding wins. The sequential
differences are one to a few rounds on five seeds, which is noise at this
sample size; the ilr arm has the best medians and the worst tails (16 and 11
rounds in two runs under the neural-network oracle, where the control never
needed more than 5), and with 30 lots under the random-forest oracle it had
the tightest range (1 to 3 against the control's 1 to 8), which is a hint,
not a result. The predictive check shows trade-offs rather than a gain: the
indicators help stability and hurt viscosity, the log-ratios help
spreadability by about 15 percent and hurt stability by about 20, and pH moves for
none of them because pH in this synthetic history is close to noise (a
random forest alone cross-validates at 0.92 of the standard deviation). The
transfer condition shows the limit the reviewers predicted: when the history
never contains a lot without the legacy ingredient, an indicator has nothing
to learn from, and the information has to come from outside the history,
which is the argument for compact descriptors of the substitutes (HLB,
Hansen parameters) and the next thing to test.

**The honest limit.** The history is synthetic. Its generator decides whether
any response is a function of log-ratios or of raw fractions, so these tables
measure the generator at least as much as the encodings. What this release
adds is the machinery, tested and run once, so the question can be asked of a
measured history in an afternoon. If you have one, see the README.

## Known limitations, openly

Small-data GPs in many dimensions; independence assumption in the joint
probability; uncalibrated feasibility classifier; random search rather than
acquisition-optimized proposals; Monte Carlo robustness limited by model
quality; no units/test-method schema yet (planned). If you are a formulator
or statistician and see something worse than what's listed here, please open
an issue — that is exactly the feedback this project wants.
