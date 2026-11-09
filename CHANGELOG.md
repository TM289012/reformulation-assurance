# Changelog

## v0.12.9 (2026-11-09)

- **Boundary-aware encodings, run as an ablation.** Two reviewers on the Materials Science Community Discourse pointed out that the hard part of a reformulation's input space is the boundary where an ingredient goes from present to absent, which raw mass fractions under a stationary kernel treat as a very low dose, and asked for a presence-indicator-plus-dosage encoding and isometric log-ratio coordinates to be tested against the current kernel rather than swapped in. `composition_features.py` adds both encodings (`presence`, `presence_ilr`, with multiplicative zero replacement and a sequential binary partition that puts the reformulation slot first), `MixedFeatureEncoder` and `_fit_response_model` take an optional `CompositionSpec`, and the app's own fits are unchanged: the default path is raw fractions, byte for byte, which a test pins.
- **The benchmark grew two arms and a sparse-history condition.** `backtest_ranker_vs_baybe.py --arms ours,ours_presence,ours_presence_ilr,random` compares the encodings on the same oracles, pools and seeds as the BayBE comparison; `--history-size 30` starts every arm from a 30-lot subsample drawn per seed; `--output` keeps ablation rows out of `backtest_results.csv`. `ablation_boundary_features.py` adds the predictive side: repeated 5-fold cross-validation and a regime-transfer condition (trained on legacy lots, predicting legacy-free lots), scored by RMSE, negative log predictive density and interval coverage.
- **The result, stated plainly in MODELING_NOTES.** On the synthetic cosmetics history no encoding wins: the sequential differences are one to a few rounds on five seeds, the predictive check shows per-response trade-offs (log-ratios help spreadability and hurt stability, indicators the reverse for stability and viscosity), and pH is close to noise for all three. The transfer condition shows the limit the reviewers predicted: an indicator cannot learn from a history that never lacks the removed ingredient, which is the case for descriptors of the substitutes, the next thing to test. Raw rows: `backtest_ablation_results.csv`, `backtest_ablation_results_h30.csv`, `backtest_ablation_predictive.csv`. Suite: 135 tests.

## v0.12.8 (2026-11-03)

- **A way to reach the maintainer that is not a GitHub issue.** The README, the public demo's front page and the project pages now carry a plain email address (tm.0624789@gmail.com) next to the issue tracker, with the standing offer spelled out: send an old trial sheet, get back what the tool makes of it within two days. A formulator arriving from a forum should not need a GitHub account to ask a question.
- **The public demo is kept awake, properly.** The first day of the scheduled visit showed GitHub dropping most scheduled runs and one run giving up on a sleeping deployment without saying what it saw. The job now runs every two hours, waits up to eight minutes, presses the wake button again every two minutes (in any frame, with a DOM click as the fallback), presents a plain desktop Chrome user agent, holds the session for 45 seconds once the app is up, and uploads a screenshot and the page text of every visit as a workflow artifact, so a red run can be read instead of guessed at.
- **The benchmark says what its history is.** `backtest_ranker_vs_baybe.py` described the 88-lot cosmetics history as "real". It is the synthetic example that ships with the app, and the docstring, the oracle's docstring and the printed summary now say so. The benchmark question, which selection brain finds a qualifying candidate in fewer rounds from the same starting history, does not depend on the history being measured, but the wording should not have implied it was.
- README notes that the `.eln` export is listed on the RO-Crate project's tools page and ELN use-case page (added by the RO-Crate maintainers on 30 October 2026) alongside the ELN Consortium listing.
- This release also carries v0.12.7 (2 November), which went to the main branch without a tag: the food example for the Red No. 3 deadline, the demo catalog, the sandbox that preloads both projects and opens on `?demo=food` or `?demo=cosmetics`, the experiments-on-record table, and the walkthrough page. Suite: 117 tests.

## v0.12.7 (2026-11-02)

- **A food example: replacing FD&C Red No. 3.** A third built-in dataset, `demo_food_color_replacement.csv`, is a strawberry gummy colored with Red No. 3, whose food use ends on 15 January 2027. 90 synthetic lots: 18 legacy lots that define the color standard and 72 trials of a black carrot anthocyanin extract and a beet juice concentrate across cook temperatures and hold times, 8 of them failed (color lost during a hot cook, an earthy off-note, a deposit that did not set). Specifications on color difference from the standard (at most 3.0), redness (a* at least 38), color retention through a four-week storage test (at least 85 percent) and a panel flavor score (at least 7.5). The generator, `make_demo_food_color.py`, documents the simple model behind every number and regenerates the file from a fixed seed.
- **One catalog for the example projects.** `demo_catalog.py` now holds the three datasets (coatings, cosmetics, food) with their configurations, and **New project, Start with demo** offers a choice of example instead of the coatings project alone; the button is now *Create demo project*. `demo_seed.py` and `app.py` both read the catalog, so the public sandbox and the local install cannot drift apart.
- **The public sandbox preloads both the cosmetics and the food project**, and completes an older sandbox in place: a redeploy that ships a new dataset adds the missing project to the existing demo workspace instead of waiting for the next wipe, and still never writes into any workspace that is not the demo's own. A link of the form `?demo=food` (or `cosmetics`) opens the sandbox on that project, so a post aimed at food developers lands on the Red No. 3 example. `ProductStore.find_user_by_email()` is the plain lookup this needs; it is not a sign-in attempt and cannot lock the account.
- **Project overview lists the experiments on record**, every imported lot and every experiment run in the tool with its status and notes, sortable, so the failure reasons in a history can be read without exporting it.
- A walkthrough for the food case, [red3-replacement.html](https://tm289012.github.io/reformulation-assurance/red3-replacement.html), joins the project pages. Suite: 117 tests.
- **The public demo is kept awake.** Community Cloud sleeps an app after a stretch without visitors and shows a "Zzzz" page; the first expert who followed a forum link to the demo met that page and left. A scheduled GitHub Action (`.github/workflows/keep-demo-awake.yml`, every four hours) now opens both public deployments in a real browser with `tools/keep_demo_awake.py`, presses the wake button when it is shown, and waits for the app to render, so the link in a post leads to the app and not to a sleeping page.

## v0.12.6 (2026-10-02)

- **`.eln` export declares ELN file format `1.2+20260923`.** The ELN Consortium's merged test suite (TheELNFileFormat #159) names the September 2026 version `1.2+20260923`, so the root Dataset's `conformsTo` and the specification `Profile` node now use `https://purl.archive.org/purl/elnconsortium/eln-spec/1.2+20260923` in place of the shorter `1.2+202609`. A crate with the old identifier is read by the suite as plain RO-Crate 1.2 and skips the consortium's own checks. Nothing else in the archive changes.

## v0.12.5 (2026-09-28)

- **Chunky-data rule, after Donald Wheeler.** Identical or near-identical replicate readings do not measure repeatability; they show only that the variation is smaller than the recording step can resolve. Following Donald J. Wheeler's "What is Chunky Data?" (Quality Digest Daily, 7 December 2011), each replicate group goes through his three steps: determine the recording step, compute the moving-range chart's upper limit (3.268 times the average moving range) from the group's own readings in run order, and count the possible range values within it, zero included. Three or fewer is chunky; four is his borderline-safe condition and is judged normally. The counting reproduces the paper's own figures (19 possible values in Figure 3, 2 in Figure 4, 4 at the borderline in Table 3).
- **Recording steps.** A new setting on the Qualification page records the step each response is logged in (0.01 for pH read to two decimals, 50 for a viscosity logged in steps of 50). When it is not set, the step is read off the numbers, erring toward the coarser step: the finest decimal place used, then the largest step of 1 or 5 times a power of ten that every reading shares. Stored numbers drop trailing zeros (pH 7.00 reads as 7), so setting the step is the reliable route; each note says which step was used and where it came from.
- **The CV limit and round-off.** Round-off biases estimates of spread (Wheeler shows a standard deviation first inflating, then collapsing toward zero, and hiding the problem), badly so for chunky data. The gates therefore apply each CV limit to the largest CV the readings allow at their recording step, a bound that holds for the true values whether the readings were rounded or truncated, provided the step is right. For finely recorded data it is barely above the CV (pH 5.42, 5.39, 5.36: 0.67% against 0.56%); for coarse readings it is what decides, so scores of 9, 8, 9 on a whole-number scale no longer pass an 8% limit on a CV of 6.7% when their true CV could be 12.9%. A group that fails only on the bound is told so, with the advice to record one more digit where the instrument or scale allows one; more replicates will not reliably fix chunky data. The bound is the tool's own safeguard, not a formula from the paper.
- **The consistency screen and round-off.** The screen computes XmR limits from the other replicates; an odd replicate outside them means the formulation is not yet reproducible. Where a replicate's siblings are chunky (two identical readings, say), limits built from them would be deflated by round-off and give the false alarms Wheeler warns about, so they are widened by the most the round-off could hide; a value outside even those limits is inconsistent however the readings were rounded. This also applies inside chunky groups, so 5.40, 5.40, 5.43 passes while 5.40, 5.40, 5.90 and a 5.8 among five readings of 5.4 are flagged. The widening is also the tool's own. Replicates are taken in replicate order (R1, R2 ... R10), not text order.
- **Browser tools.** The Baseline Drift Checker draws a plain running record with no limits on chunky data, refuses to call signals, and passes on Wheeler's advice: record the extra digit even if it seems uncertain, space readings further apart, consider a different measurement system if the current one cannot give another digit, and treat chunky counts as a running record only. Its small-sample floors are corrected (an end point can signal from 5 values, not 6). The Replicate Noise Checker judges each formula at its own recording step and declines to compare a difference against a noise estimate the step has hidden. Both read decimals in scientific notation and infer the step the same way as the app.
- `measurement_increment()`, `possible_range_values()`, `chunky_data_check()`, `cv_upper_bound()`, `declared_recording_steps()`, and `chunky`, `possible_values` and `widened` in `wheeler_screen_detail()` in `assurance_v4.py`; `chunky_<response>` and `cv_upper_<response>` columns in the replicate summary, and replicate numbers in the notes taken from the run numbers. With 3 to 6 readings the count of possible values is rough and depends on run order, which is why a chunky call changes how a group is judged rather than failing it outright. Non-finite readings are ignored, and a group missing a response is no longer mistaken for a chunky one. Suite: 110 tests.

## v0.12.4 (2026-09-27)

- **`.eln` export on RO-Crate 1.2 with the ELN 1.2+202609 declaration.** The metadata descriptor declares RO-Crate 1.2, the root Dataset declares both the RO-Crate profile and the ELN specification (as `Profile` entities), and the crate is a fully flattened JSON-LD graph: `variableMeasured` now holds references and the `PropertyValue` nodes are top-level entities, which is what RO-Crate requires and what the ELN Consortium's updated checks enforce. The root also names the exporting workspace as `publisher` and carries a licence entity; the author has a local identifier and an `affiliation`. Checked against the consortium's new test suite (archive structure, metadata rules, schema, `rocrate` parse: all pass at version 1.2+202609).
- Known consequence: eLabFTW's importer resolves `variableMeasured` references only for crates carrying eLabFTW's own internal `version` marker, so the extra-fields panel is not populated on import until that is fixed upstream (reported); the entry, tags, body and every attachment import as before, and the same values are in the body.

## v0.12.3 (2026-09-27)

- **Mail failures are explained where the administrator looks.** The Team page's outbox now shows why a message failed in plain language next to its status (a rejected app password, a missing username, a wrong port, a blocked or unreachable host), lists how many messages failed since the last successful send, and offers a *Retry failed messages* button once the settings are fixed. Failed password resets are counted without exposing their recipients or links.
- **Send a test email.** One click on the Team page sends a short message to the signed-in administrator's own address through the outbox, so a new configuration can be checked without inviting anyone.
- **Implicit TLS (port 465) is supported.** `REFORMULATION_SMTP_PORT=465` now connects with TLS from the first byte automatically; `REFORMULATION_SMTP_SSL` overrides the choice. Values pasted with stray spaces or newlines are trimmed, and a non-numeric port is reported on the Team page instead of stopping the app.
- The invitation and roster forms say what actually happened: emailed, or created but not emailed (with the reason), or created on an instance without mail. The password-reset form's wording depends only on the instance's configuration, never on whether the account exists.
- Suite grows to 88 with `tests/test_mail_delivery.py` (transport selection, error translation, retry from the outbox, admin-only diagnostics).

## v0.12.2 (2026-09-27)

- **Mail goes out immediately when SMTP is configured.** Password-reset links and invitations (single and roster) are now delivered at the moment they are created instead of waiting for an admin to open the Team page. Without SMTP, behaviour is unchanged: messages wait in the outbox.
- `hosted.py`: a second entry point for Streamlit Community Cloud, which allows one app per repository, branch and file; the public demo keeps `app.py`, a hosted workspaces instance deploys from `hosted.py` (documented in DEPLOY.md).

## v0.12.1 (2026-09-27)

- **Replicate screen: identical sibling readings no longer flag a value one step away.** In v0.12.0 the leave-one-out screen treated identical readings among the other replicates as proof of zero variation, so pH 5.4, 5.4 then 5.3 was reported as "replicate #3 inconsistent". Identical readings only show variation below the instrument's resolution; limits built from them have no width and cannot judge anything. Such values are now reported as unjudged by the screen (the note says so) and are left to the CV check, which still fails a genuinely wild value. Spread among the siblings still catches an outlier as before. Found while preparing the example archive for the ELN Consortium listing.

## v0.12.0 (2026-09-27)

The hosted release: the same app can now serve several workspaces from one PostgreSQL database, so a lab, a brand or a course can use it without installing anything, while the local SQLite install stays the default and is unchanged. Nothing about the science moved; this release is plumbing plus the two features a course section needs, and it is documented in `DEPLOY.md` and the new for-courses page.

- **PostgreSQL backend.** `db_backend.py` sits under the stores and speaks either SQLite (a file path, the default) or PostgreSQL (a `postgresql://` URL via psycopg 3): placeholder and DDL translation, portable upserts, dict/index row access, and one shared server connection per process that is reused across requests, serialized with a lock, and reopened when a serverless database has suspended it. `REFORMULATION_DATABASE_URL` selects it.
- **Self-serve workspaces.** With `REFORMULATION_OPEN_SIGNUP=true` the sign-in screen gains a *Create a workspace* tab. A new email gets its own workspace as owner; an existing account can add a workspace by giving its password; a wrong password gets a clear refusal instead of a database error.
- **Encrypted exports without a disk.** On PostgreSQL the encrypted dossier artifacts are stored as ciphertext rows instead of files, and the app reads the Fernet key from `REFORMULATION_ARTIFACT_KEY` (owners see a red banner while it is missing on a hosted deployment, because a key generated on a disposable disk is a data-loss bug waiting to happen).
- **Sign-in throttling.** Eight failed attempts lock an email for fifteen minutes; a successful sign-in clears the counter. Local installs get this too.
- **Roster invitations.** The Team page takes a pasted list of addresses and creates one invitation link each (two-week validity by default), with the links as a table and a CSV for a course announcement; malformed addresses are reported, not fatal.
- **Workspace overview.** A new admin page lists every project in the workspace with who started it, imported lots, planned and completed experiments, batches, gates passed, the qualification progress score, live approvals, dossier exports and last activity, with a CSV export. Built for instructors and team leads. `docs/for-courses.html` describes the two-hour guided lab, section setup and a rubric.
- **`.eln` export aligned with the ELN file format's September 2026 wording** (1.2+202609): `keywords` is a comma-separated string, every attached file is listed in the root `hasPart` import list alongside the experiment entry, the metadata descriptor carries its `version`, and empty evidence tables are no longer attached as zero-byte files (they stay in `SHA256SUMS.txt`, and the crate description names them). The crate stays on RO-Crate 1.1 until the consortium's validator supports 1.2. Checked against the consortium's own test suite (archive structure, ro-crate-py parse, metadata rules, JSON schema: all pass) and eLabFTW's current importer source.
- Settings can come from Streamlit secrets as well as the environment (`REFORMULATION_DATABASE_URL`, `REFORMULATION_OPEN_SIGNUP`, `REFORMULATION_PUBLIC_URL`, `REFORMULATION_ARTIFACT_KEY`, `REFORMULATION_DEMO_MODE`).
- The SQLite-only operations (verified backup files, the PostgreSQL migration bundle) hide themselves on PostgreSQL, where the database provider's backups apply.
- Testing: the full suite runs unchanged against PostgreSQL with `REFORMULATION_FORCE_DATABASE_URL` (each test store in its own schema); `tests/test_app_smoke.py` drives the real `app.py` headlessly through Streamlit's `AppTest` (first-owner bootstrap, sign-in, open sign-up, mismatched passwords). Suite grows to 74.

## v0.11.0 (2026-09-05)

The notebook release: the qualification dossier now exports as a `.eln` archive, the ELN Consortium's open format (a zipped RO-Crate), so a project's evidence can be filed in the electronic lab notebook a lab already keeps. Prompted by the eLabFTW community discussion on the best import path, where maintainer Nicolas CARPi recommended `.eln` while noting that a full JSON export is worth keeping. Both are here.

- New **Export to your lab notebook (.eln)** on the Approvals & dossier page. One file, one experiment entry: a compact HTML summary as the entry body (gates, approvals, evidence hash), and every evidence table, the printable dossier, the scientific-evidence JSON, the signed evidence snapshots, the checksum list and (optionally) the Excel workbook as attached files. Each attachment is described in the crate metadata with its media type, size and SHA-256.
- The entry carries structured extra fields for eLabFTW (project, dossier version, evidence hash, qualification progress, gates passed, generated by/at, software), so the evidence hash is visible beside the entry, not buried in an attachment.
- `scientific_evidence.canonical.json` ships alongside the readable JSON: `sha256sum` of that one file reproduces the evidence hash recorded in the metadata and in every signature.
- Generating an `.eln` records a dossier version exactly as the ZIP export does, so the notebook entry and the dossier history agree on version numbers, and the export is recorded in the audit trail (`eln_exported`).
- `eln_export.py` doubles as a CLI: `python eln_export.py --db <workspace.db> --project <id> --email <you> --out project.eln`.
- Verified against the ELN file format specification, the source of eLabFTW's importer (root dataset → `hasPart`, `genre: experiment`, `temporal`, `keywords`, inline `elabftw_metadata`, per-file `sha256`), and the reference RO-Crate parser (`ro-crate-py`), which reads the archive and round-trips all entities. A live import into a running eLabFTW instance is the remaining check.
- Test suite grows to 52.

## v0.10.0 (2026-09-01)

Honest small-sample behavior for judging repeatability from three to six replicates, with thanks to Donald J. Wheeler for his advice. The confirmation gate now looks at the running record first, judges a suspect value against limits computed from the other replicates, and only then lets the CV mean anything.

- Replicate consistency screen: every replicate group is screened before its CV counts. Each value is judged against natural limits (mean ± 2.66 × average moving range) computed from the other replicates in run order; a group with an inconsistent replicate cannot pass a gate, and the gate's remaining-requirements text names the odd replicate. With 3-6 replicates the screen is indicative, not definitive, and says so.
- Running records in the app: the Qualification page gains a "look before computing" expander that plots each replicate group's values in run order, per response.
- Drift checker honesty at small n: XmR limits cannot flag the first or last point below 6 values or a middle point below 8, and limits only stabilize around 17+. The checker now says "too few points for this chart to say anything" instead of issuing a hollow all-clear below those floors, and labels 6-16 point verdicts as approximate.
- Replicate Summary gains `consistent_<response>` and `screen_note_<response>` columns.
- Test suite grows to 45.

## v0.9.0 (2026-08-14)

The attach release: the analysis goes back to where formulators actually live. Prompted by feedback from a practicing formulator that a tool like this belongs as a feature of a bigger program rather than a standalone destination — and for this audience, the bigger program is the spreadsheet.

- One-click Excel workbook export on the Approvals & dossier page: a single .xlsx with the whole project as tabs — experiments, recommendation batches, qualification gates, replicate summary, run- and formulation-level calibration (summaries and the frozen-prediction observations), robustness runs, approvals, and the audit trail, with a Read Me cover sheet carrying the scientific evidence SHA-256.
- Workbook exports are recorded in the audit trail.
- Landing page and README now lead with the attach-to-your-spreadsheet story: CSV or Excel in, files you keep out, no lock-in.

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
