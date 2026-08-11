# MASTER PLAN — ML4H 2026 Findings submission

**Deadline:** Sep 10 2026, 11:59pm AoE. Target internal submit date: **Sep 7** (3-day buffer).
**Today:** Aug 11 2026. **College exams end Aug 24** — Tier 0 fits around exams, Tier 1 compute runs unattended during them.

Paper: `C:\Users\Asus\Downloads\masters\LATEST_MANUSCRIPTS\ML4H_Findings\main.tex`
Code/results: `D:\Projects\diabetes_prediction_project\federated\`
BRFSS source (verified present, 54MB): `C:\diabetes_prediction_project\data\03_processed\brfss_final.csv`

---

## 0. MONEY — resolved from the actual ML4H email thread (verified 2026-08-11)

| Item | Verdict | Source |
|---|---|---|
| Submission fee | **None.** Nothing to pay now. | CFP + FAQ + full email thread — no fee mentioned anywhere |
| Registration if accepted | **Required.** "You must pay a registration fee for your paper to be part of the ML4H conference." | Sasha (ml4h@ahli.cc), 2026-08-02 |
| Amount | **$125–250 USD, not finalised** | Sasha, 2026-08-03 |
| Reduced rate | Exists, but **not finalised, case-by-case, apply only AFTER acceptance** | Sasha, 2026-08-05 |
| Travel to Sydney | **Likely avoidable** — "in certain circumstances, virtual, asynchronous poster presentations may be available" | Sasha, 2026-08-07 (newest email) |
| Findings publication fee | None — non-archival | Sasha, 2026-08-03 + FAQ |

**Bottom line: ₹0 today. Realistic worst case ~$250 in Nov/Dec, plausibly less after reduced-rate application, and the Aug 7 email means you probably do not need to fly to Australia.** Decide the fee question in late October when you know the decision and the rate — not now.

---

## 1. VERIFIED DEFECTS (found by reading result files + scripts, not by reading the paper)

### D1 — [BLOCKER] The decisive control is evaluated on a different, non-randomly filtered population
The sample-matched centralised model — the control that carries the paper's headline claim — is evaluated on **720,161** BRFSS respondents. Every other model is evaluated on **1,282,897**.

Root cause (confirmed by reading both scripts):
- `07_external_validation.py:158–165` (all main models): `dropna(subset=['DIABETES'])` then `fillna(median)` / `fillna(mode)` → **imputes, keeps 1,282,897**
- `run_matched_subset_external.py:76` (matched control): `df_b[available].dropna()` → **complete-case deletion, keeps 720,161**

This is worse than "a different draw." It is a different *missing-data policy*, dropping 562,736 respondents (43.9%) non-randomly — complete-case respondents are systematically healthier and better-surveyed. The 0.024 gap is therefore not comparable to the 0.031 and 0.052 it is being compared against, and that comparison is the paper's central claim.

**Fix:** rewrite `run_matched_subset_external.py` to import the identical cleaning path as `07_external_validation.py`. Best: extract the BRFSS load+clean into one shared function in `data_utils.py` and call it from both, so the two can never diverge again. ~30 min. Then rerun.

### D2 — [MAJOR] Platt-transform bug produces a self-contradicting result file
`cmpb_experiments.py:238–240` applies stored Platt coefficients to `logit(p)`, but `17_fedavg_calibration.py:95` fit them on raw `p` (`LogisticRegression().fit(p_cal_raw.reshape(-1,1), y_cal)`).
Consequence: `fedavg_calibration_results.json` says Platt ECE = **0.0072** (correct, this is what the paper cites), `ece_cis.json` says **0.1105** for the same model and data. A reviewer cloning the repo gets two contradictory numbers.
**Fix:** `sigmoid(coef * p + intercept)` — drop the logit. Rerun. ~15 min.
Also: the isotonic CI in that file is not a real bootstrap (the code comment admits it). Either store the fitted `IsotonicRegression` and bootstrap it properly, or drop that CI.

### D3 — [MAJOR] Zero seed control anywhere
No `seed` / `random_state` / `torch.manual_seed` in `03_federated_simulation.py`, `02_centralised_baseline.py`, `08_scaffold_baseline.py`, `00_config.py`. Every headline number is one uncontrolled run. Model is 4-layer MLP, **4,993 params** — 5 seeds is minutes of compute, not hours.

### D4 — [MAJOR] Race is a model input but never audited
Node CSV columns: `RIDAGEYR, RIAGENDR, RIDRETH3, BMXBMI, SMOKING, PHYS_ACTIVITY, HEART_ATTACK, STROKE, DIABETES`. **`RIDRETH3` is race/ethnicity — a model feature.** Fairness section reports age, sex, BMI. Never race. On a fairness paper, at a health-equity venue, for a disease with documented US racial disparities, this is the first question a reviewer asks.
Partial prior work exists (`fairness_comparison.json` has race EOD) but it is **internal NHANES only** and the "federated" model there is not clearly FedAvg — do not copy it into a paper whose headline fairness numbers are FedAvg-on-BRFSS. Needs a proper external run.

### D5 — [MAJOR] Survey design ignored
`nhanes_merged.csv` contains `WTINT2YR, WTMEC2YR, SDMVPSU, SDMVSTRA` — the NHANES survey weight / PSU / strata variables. They are dropped from the node files and never used. BRFSS is likewise a raked complex survey. All inference is unweighted with ordinary bootstrap/DeLong.
Supporting literature verified real via arXiv API: `arXiv:2605.08963` (Survey-aware Machine Learning) and `arXiv:2603.28320` (Design-Based Inference for the AUC with Complex Survey Data).

### D6 — [MAJOR] Hardcoded absolute path breaks reproducibility
Both `07_external_validation.py:48` and `run_matched_subset_external.py:23` hardcode `C:\diabetes_prediction_project\data\03_processed\brfss_final.csv`. A reviewer cloning the anon repo cannot run anything. Move to a PATHS block / env var in `config_paths.py`.

### D7 — [MINOR] NHANES cohort accounting unexplained
4,500 + 3,406 + 4,000 = 11,906, but NHANES n is stated as 15,650. The missing 3,744 are never accounted for. One sentence + a cohort-flow line fixes it. Without it, a reviewer suspects leakage or undocumented filtering.

### D8 — [MINOR] Literature gap asserted, not cited
"Most clinical-FL work reports discrimination on a single cohort and stops" — uncited. General fairness-in-FL work (RESFL, FinP, unified fair FL) exists and is not differentiated.

### D9 — [MINOR] The ~40,000-per-node DP claim is not backed by an experiment
`dp_results_extended.json` sweeps ε, not cohort size. Either run a size-scaling sweep or soften the claim to what was measured.

---

## 2. CHATGPT'S REVIEW — what to take, what to discard

**Take (verified correct and valuable):**
- The 720,161 discrepancy → real, and the mechanism is worse than it said (D1).
- **Client-weighted centralised control** → the single best scientific idea in its response. Genuinely sharpens the paper.
- Multi-seed, multiple partitions, leave-one-client-out, SCAFFOLD-same-optimizer, direct bootstrap Δ test, AUPRC, NHANES accounting, survey weighting → all sound.
- "Do not add more models" → correct.
- Its refusal to promise a guarantee → correct.

**Discard / correct:**
1. **"Submit to Proceedings and let it fall back to Findings" — DO NOT DO THIS.** Proceedings is archival and the CFP states archival submissions "cannot be already published in or simultaneously under review at any other archival venue." Your own email to Sasha says this work is "a short version of work currently under review at a journal" (CMPB). A Proceedings submission would be a **dual-submission violation** — an integrity problem with both ML4H and CMPB, not a strategy choice. ChatGPT missed this despite citing the CFP. Findings is not the safe fallback here; it is the only permissible track.
2. **It hallucinated your title and abstract.** It quotes the title as "Federated Learning's Generalisation Advantage Is Largely Data Composition — But Its Fairness Advantage Is Not" and an abstract phrase "the fairness advantage, however, survives." Neither is in `main.tex`. Actual title: "Beyond Privacy: Auditing Generalisation, Fairness, and Calibration in Federated Diabetes Risk Prediction." Several of its recommendations (esp. #25, #26) are built on that invented framing — ignore those.
3. **It overstated code sharing as mandatory.** CFP: code sharing is *encouraged*; the *statement* is what is required — "If you are not sharing code, you must explicitly state this in the paper." So a missing anon repo is **not** an automatic desk reject. Still do the repo, but it is not a submission blocker.
4. **It missed the Aug 7 email** on virtual asynchronous posters, and left the sole-author presentation problem unresolved when the organisers had already resolved it.
5. **Its 30-item list is too large for 4 pages and 30 days**, most of it competing for space that does not exist. Appendices are unlimited — that is where the bulk goes.

---

## 3. THE PLAN

### TIER 0 — Safety. Do now, before any science. (~3 hrs)
- [ ] **T0.1 Reciprocal-reviewer exemption email to ml4h@ahli.cc — TODAY.** Highest-risk item on this list and it has nothing to do with the paper. No qualifying prior archival publication → must request the first-time-author exemption. Desk-reject risk if unresolved.
- [ ] T0.2 Fix D1 (shared BRFSS clean function) — correctness, not enhancement.
- [ ] T0.3 Fix D2 (Platt transform) + regenerate `ece_cis.json`.
- [ ] T0.4 Fix D6 (PATHS block, no hardcoded C:\ paths).
- [ ] T0.5 Add `SEED` to every training script (D3 prerequisite).
- [ ] T0.6 D7 one-line cohort accounting in Methods.
- [ ] T0.7 Anonymise repo (file list + scrub terms already specified in chat), scrub PDF metadata (`pdfinfo` — MiKTeX can embed OS username even when the author block says Anonymous).

### TIER 1 — The three that move the paper. Run after T0.2 lands. (~1 day compute, unattended)
- [ ] **T1.1 Every model → identical BRFSS population.** Direct consequence of T0.2. Re-run all seven models, regenerate Table 1.
- [ ] **T1.2 5 seeds × all models.** Report mean ± SD. Point estimates stay in Table 1; per-seed table → appendix.
- [ ] **T1.3 Client-weighted centralised control.** Weight each client's samples to reproduce FedAvg's effective aggregation weighting. This is the experiment that converts a speculative mechanism into evidence. Three outcomes, all publishable:
  - reproduces FedAvg's fairness gain → *the fairness advantage is weighting, not federation* (best paper)
  - does not reproduce it → *federated optimisation itself contributes* (also strong)
  - ambiguous → report honestly as inconclusive
- [ ] **T1.4 Δ-statistics with CIs.** Bootstrap `Δgap = gap(FedAvg) − gap(matched)` and `Δfairness` directly, paired on the same test set. Replaces the current weaker "non-overlapping CIs" argument.

### TIER 2 — High value, appendix-friendly. (~2–3 days)
- [ ] T2.1 Race-stratified external fairness, FedAvg on BRFSS (D4).
- [ ] T2.2 SCAFFOLD with the same optimiser as the others → separates aggregation effect from optimiser confound. Currently the paper admits this confound and cannot resolve it; one run resolves it.
- [ ] T2.3 Leave-one-client-out (esp. elderly-rural) → direct mechanistic test.
- [ ] T2.4 2–3 alternative partitions (not ChatGPT's 5) → robustness to federation construction.
- [ ] T2.5 Survey-weighted external evaluation (D5) + cite the two verified arXiv refs.
- [ ] T2.6 AUPRC alongside AUROC, overall + per subgroup.
- [ ] T2.7 Add the 2–3 missing citations (D8).

### TIER 3 — Only if Tier 0–2 are done and there is time.
- [ ] T3.1 Calibration slope + intercept, subgroup calibration.
- [ ] T3.2 Decision curve analysis / net benefit.
- [ ] T3.3 Threshold sensitivity (0.20 / 0.30 / 0.35 / Youden / 0.50) — defends the "31% fewer false positives" claim against a post-hoc-threshold accusation.
- [ ] T3.4 DP cohort-size scaling sweep, or soften the 40,000 claim (D9).
- [ ] T3.5 `reproduce_all.sh` + per-table/per-figure scripts.

### Rules while executing
- **Every claim in the paper must map to a result file.** Build the claim→evidence table before writing.
- **The story must be allowed to change.** If T1.3 shows the fairness advantage is weighting, that is the better paper — take it. Do not force the current framing.
- **Appendix is free.** Main text stays 4 pages; robustness lives in the appendix.
- Do not add more models.

---

## 4. TIMELINE

| Window | Work |
|---|---|
| Aug 11 (today) | T0.1 exemption email. T0.2–T0.5 code fixes. |
| Aug 12–23 (exams) | Tier 1 compute runs unattended. Repo anonymisation in spare hours. |
| Aug 24–31 (bandwidth returns) | Read Tier 1 results, decide whether the story changes, run Tier 2. |
| Sep 1–5 | Rewrite paper around final results. Regenerate figures/tables. Appendix. |
| Sep 6 | Full verification pass: numbers cross-checked at every occurrence, Overleaf official-template compile, PDF metadata scrub, anon link live. |
| **Sep 7** | **Submit.** |
| Sep 8–10 | Buffer only. Do not plan to use it. |

---

## 5. ACCEPTANCE ODDS — honest position

No published ML4H Findings acceptance rate exists for 2025 or 2026 (searched OpenReview, CFPs, past proceedings). Any specific percentage — mine or ChatGPT's 45–60% / 75–85% — is an unsourced guess dressed as a statistic. Treat them as such.

What is defensible:
- **The track's stated bar is discussion value, not technical sophistication.** ML4H's reviewer guidelines say a Findings paper "may not necessarily meet the same level of technical sophistication... as long as it meets the criteria of generating productive and interesting discussions," and explicitly solicit negative results and critiques. This paper's audit framing is a strong fit — ChatGPT's 45–60% likely *understates* the current position for this specific track.
- **The largest current risk is not scientific.** It is (a) the reciprocal-reviewer requirement, and (b) D1 — a reviewer noticing that the decisive control was evaluated on a differently-filtered population. Both are fully fixable and both are cheap.
- **After Tier 0 + Tier 1, the paper stops having a findable methodological hole.** At that point residual risk is reviewer variance and competition, which no amount of work removes.

The honest summary: you cannot make a paper unrejectable. You *can* make it so that a reviewer who wants to reject it has to argue about taste rather than point at an error. Tier 0 + Tier 1 gets you there. Tier 2 makes it hard to argue about taste either.
