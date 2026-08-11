# FINDINGS LOG — ML4H 2026 rework

Running record of executed work and results. Append-only; newest at bottom.
Every number here traces to a file in `federated/results/`.

---

## 2026-08-11 — T0.2 / T0.3 / T0.4 complete + T1.2 (partial)

### T0.4 — BRFSS path resolution fixed
`config_paths.py` now resolves `BRFSS_PATH` via `_resolve_brfss_path()`: env var
first, then a candidate search. Two scripts previously hardcoded
`C:\diabetes_prediction_project\...`, which made the repo unrunnable for anyone
else. The config default pointed at a `D:` path that does not exist — surfaced
immediately when the fixed script was first run.

### T0.3 — Platt-transform bug fixed
`cmpb_experiments.py::platt_calibrate` applied stored Platt coefficients to
`logit(p)`, but `17_fedavg_calibration.py:95` fit them on raw `p`. Fixed to
`sigmoid(coef * p + intercept)`. Regeneration of `ece_cis.json` still pending.

### T0.2 / T1.1 — BRFSS pipeline unified. **This changed the paper's headline result.**

`data_utils.load_brfss_clean()` is now the single BRFSS entry point, a faithful
extraction of `07_external_validation.py` (which produced the paper's verified
main-model numbers, so those are unchanged). `run_matched_subset_external.py`
now calls it instead of rolling its own.

Two defects removed from the matched control:
1. complete-case deletion → 720,161 rows, vs 1,282,897 for every other model
   (43.9% dropped, non-randomly — complete-case respondents are systematically
   healthier and better surveyed)
2. no BRFSS 0/1 → NHANES 1/2 sex remap — a miscoded feature fed to a model
   trained on 1/2 (BRFSS `Gender` confirmed 0/1)

**Result of the fix** (`results/matched_subset_sensitivity.json`, seed 42):

| | before (buggy) | after (fixed) |
|---|---|---|
| n_brfss | 720,161 | **1,282,897** ✓ matches all models |
| external AUC | 0.7712 | 0.7514 |
| generalisation gap | **0.0239** | **0.0476** |

### T1.2 (matched control only) — 5 seeds confirm it is not seed noise

`32_multiseed_matched_control.py` → `results/matched_control_multiseed.json`

| seed | internal | external | gap |
|---|---|---|---|
| 42 | 0.7990 | 0.7514 | 0.0476 |
| 43 | 0.8042 | 0.7523 | 0.0519 |
| 44 | 0.7966 | 0.7485 | 0.0481 |
| 45 | 0.8113 | 0.7571 | 0.0543 |
| 46 | 0.7947 | 0.7526 | 0.0421 |

**gap = 0.0488 ± 0.0047, range [0.0421, 0.0543]**
**external AUC = 0.7524 ± 0.0031**

The old 0.0239 lies far outside the full range. The difference was caused by the
defects, not by initialisation variance.

### CONSEQUENCE — the manuscript's central caveat is now false as written

Corrected gap ordering (matched control multi-seed; all others still single
unseeded runs and therefore provisional):

| model | gap |
|---|---|
| XGBoost (centralised) | 0.069 |
| DiabetesNet (centralised, n=15,650) | 0.052 |
| **Matched centralised (n=11,906)** | **0.0488 ± 0.0047** (was 0.024) |
| FedAvg | 0.031 |

FedAvg's gap is now **smaller** than the sample-matched control, not larger.
The paper currently claims the opposite. Three passages are invalidated:

- Abstract: "A sample-matched control shows that data composition explains part
  of the gap advantage"
- Results: "a sample-matched centralised control ... reaches a gap of 0.024,
  *smaller* than FedAvg's: data composition, not federation alone, explains part
  of the advantage. Federation's distinctive value therefore lies less in the
  gap itself than in privacy, fairness, and regulatory feasibility."
- Discussion: "federation's edge over a *sample-matched* centralised model rests
  on privacy, fairness, and regulatory feasibility rather than on the
  generalisation gap alone"
- Table 1, "Matched centralised*" row: 0.795 / 0.771 / 0.024

**DO NOT REWRITE YET.** FedAvg's 0.031 is still a single uncontrolled run. The
comparison is only defensible once every model is multi-seeded (T1.2 full). The
direction may hold, narrow, or reverse again. Rewrite after T1.2 + T1.3.

### Strategic note
The correction moves the result *toward* federation, which removes the
self-critical framing that was the paper's distinctive hook. The replacement
hook should come from T1.3 (client-weighted centralised control) — the
mechanistic question of whether any FL advantage is federation or merely the
implicit reweighting from client-level aggregation. That is now the paper's
most important experiment, not a nice-to-have.

---

## 2026-08-11 (later) — T1.2 FULL + T1.3 (first pass). Three more paper claims overturned.

`fl_core.py` created — one canonical seed-aware FL implementation (FedAvg /
FedProx / FedNova / equal-client-weight), extracted from 03_federated_simulation.py
so there is not a second copy of FedAvg in the project. `33_multiseed_all_models.py`
trains every model fresh per seed and scores all of them on the identical
1,282,897-row BRFSS population.

Internal AUC is now measured on held-out node validation splits. The previous
`eval_global` path scored on the full centralised NHANES pool, which **contains
the nodes' own training rows** — optimistic for federated models. Both are stored
(`internal_val`, `internal_pool`); `internal_val` is the defensible one.

### Results, 5 seeds (42–46) — `results/multiseed_all_models.json`

| model | external AUC | gen. gap | fairness gap | elderly AUC |
|---|---|---|---|---|
| XGBoost | 0.7078 ± 0.0041 | 0.0724 ± 0.0127 | 0.0738 ± 0.0163 | 0.5948 |
| Centralised-full | 0.7536 ± 0.0020 | 0.0432 ± 0.0114 | 0.0648 ± 0.0062 | 0.6634 |
| Centralised-matched | 0.7524 ± 0.0031 | 0.0488 ± 0.0046 | 0.0645 ± 0.0048 | 0.6625 |
| FedNova | 0.7455 ± 0.0067 | 0.0353 ± 0.0100 | 0.0538 ± 0.0105 | 0.6519 |
| FedProx | 0.7517 ± 0.0051 | 0.0294 ± 0.0065 | 0.0533 ± 0.0047 | 0.6607 |
| FedAvg | 0.7463 ± 0.0064 | 0.0346 ± 0.0099 | 0.0531 ± 0.0110 | 0.6530 |
| FedAvg-EqualW | 0.7441 ± 0.0056 | 0.0364 ± 0.0093 | 0.0511 ± 0.0105 | 0.6508 |

### F1 — "FedAvg attains the best external AUC" is FALSE
Paired per-seed (FedAvg − Centralised-matched external AUC):
`-0.0107, -0.0037, +0.0013, -0.0033, -0.0139` → FedAvg loses in **4 of 5 seeds**.
Both centralised controls (0.752, 0.754) beat every FL variant (0.744–0.752).
The paper's 0.757 for FedAvg sits **above the maximum of the 5-seed range**
(0.7387–0.7538) — it was a fortunate single unseeded draw.
**Abstract + Results claim must be removed.**

### F2 — The generalisation-gap advantage SURVIVES, consistently
Paired per-seed (FedAvg − Centralised-matched gap):
`-0.0042, -0.0097, -0.0226, -0.0319, -0.0023` → negative in **5 of 5 seeds**.
FedAvg's gap (0.035) is genuinely smaller than the composition-matched control
(0.049). This reverses the corrected-but-single-seed reading in the earlier entry
and reverses the paper's original (buggy) claim in the other direction.

**Critical caveat that must be stated in the paper:** the gap is smaller because
internal AUC is *lower* (FedAvg ≈0.781 vs matched ≈0.801), not because external
AUC is higher. This is reduced overfitting, not better transfer. A smaller gap
combined with equal-or-worse external discrimination is a regularisation effect,
not a performance benefit, and must not be sold as one.

### F3 — The fairness advantage SURVIVES the matched control
Every FL variant (0.051–0.054) beats both centralised controls (0.0645, 0.0648)
and XGBoost (0.074). Margin ≈0.011, and the FL/centralised separation exceeds the
seed SDs. Driven by elderly AUC — but note FL's elderly AUC (0.651–0.661) is
*lower* than centralised (0.662–0.663); the gap narrows because FL's young-group
AUC falls further. **T1.5 subgroup decomposition therefore matters: this is
Case D (both subgroups degrade, gap shrinks), the weakest of the four readings.**
Must be reported honestly, not as "FL helps the elderly."

### F4 — T1.3 first pass: client weighting is NOT the mechanism
Paired per-seed (FedAvg-EqualW − FedAvg fairness gap):
`-0.0090, +0.0023, +0.0007, -0.0012, -0.0030` → sign flips, mean −0.002, smaller
than its own spread. Equal client weighting does **not** reliably change fairness.

This is informative rather than null. Under FedAvg's n_k/N client weighting the
*effective per-sample* weight is (n_k/N)·(1/n_k) = 1/N — i.e. uniform, identical
to pooled centralised training. So `Centralised-matched` already IS the
weighting-matched control: same data, same effective sample weighting, differing
only in pooled-vs-federated optimisation. FedAvg still beats it on fairness by
0.011 across all seeds. EqualW, which genuinely *does* change weighting (small
clients upweighted), barely moves the result further.

**Reading: the fairness effect is not explained by data composition (matched
control) nor by client weighting (EqualW), which points to the federated
optimisation dynamics themselves — local epochs plus periodic averaging acting
as a regulariser.** This is ChatGPT's "opposite outcome," and it is the stronger
of the two he described. It is now the paper's candidate central contribution.

### Emerging paper story (do not write until T1.4/T1.5 confirm)
> Federated averaging behaves as a regulariser. It does not improve external
> discrimination — a composition-matched centralised model equals or beats it.
> It does consistently reduce the internal–external generalisation gap and
> narrow the elderly–young AUC disparity, and neither effect is attributable to
> training-set composition or to client-level weighting. The disparity narrows,
> however, partly because performance falls in the better-served subgroup.

### Next
- [x] T1.4 — Δ statistics with CIs → done via DeLong, see next entry
- [x] T1.5 — subgroup decomposition → done, see next entry
- [x] T0.3b — regenerate `ece_cis.json` → done, see next entry
- [ ] T2.2 — SCAFFOLD under the same optimiser (not yet multi-seeded; excluded from the table above)
- [ ] Re-examine: paper's XGBoost external 0.700 vs 0.7078 ± 0.0041 here (different split + scaled features)

---

## 2026-08-11 (later still) — T1.4 + T1.5 + T0.3b complete, 10 seeds. **The paper's model choice is wrong.**

### Infrastructure
`33_multiseed_all_models.py` now caches every prediction vector to
`results/preds_multiseed/` (gitignored), so T1.4, T1.5 and every remaining Tier-2
analysis run on CPU with no retraining. `--seed-start` allows extending the seed
set without recomputing. Seeds 42–51 (n=10) are cached.

`35_delong_deltas.py` computes all Δ statistics. `test_delong.py` verifies the
DeLong implementation against sklearn (AUC exact to 1e-10) and against a real
paired bootstrap (SE within 1%, and the paired SE is 1.92x tighter than the
independence assumption, confirming pairing is credited).

### Why DeLong, and how the two variance sources combine
A paired bootstrap over 1,282,897 rows x 7 models x 10 seeds is ~10 GPU-hours;
DeLong gives the paired covariance analytically. Every seed is scored on the
*same* BRFSS rows, so:

    Var(mean Δ) = Var(shared component) + sd_seed² / n_seeds

The shared component is the DeLong variance of whatever is computed on
seed-invariant data. For external/subgroup/fairness deltas that is the whole
statistic. For the **generalisation gap only the external term is shared** — each
seed draws a different internal split, so that noise already lives in sd_seed²
and shrinks with n. An earlier version added the full per-seed DeLong variance
for the gap, double-counting the internal term and inflating its SE 2.6x
(0.0159 → 0.0060); that was wrong and is fixed.

### Δ vs Centralised-matched, 10 seeds (`results/delong_deltas.json`)

| model | Δ external AUC | Δ fairness gap | Δ gen. gap |
|---|---|---|---|
| Centralised-full | +0.0034 [+0.0009, +0.0059] p=.007 | −0.0035 p=.14 | −0.0045 p=.17 |
| **FedProx** | **+0.0000 [−0.0028, +0.0028] p=.98** | **−0.0140 [−0.0197, −0.0082] p=1.8e−6** | **−0.0117 [−0.0184, −0.0049] p=7.0e−4** |
| FedAvg | −0.0049 [−0.0098, −0.0001] p=.048 | −0.0154 [−0.0238, −0.0070] p=3.2e−4 | −0.0070 [−0.0145, +0.0004] p=.063 |
| FedNova | −0.0053 [−0.0103, −0.0002] p=.041 | −0.0148 p=5.8e−4 | −0.0068 p=.066 |
| FedAvg-EqualW | −0.0072 [−0.0120, −0.0022] p=.004 | −0.0158 p=7.9e−5 | −0.0050 p=.18 |
| XGBoost | −0.0417 p=4.9e−98 | +0.0030 p=.64 | **+0.0258 p=6.7e−13** |

Sanity check that the method detects real effects: Centralised-full beats
Centralised-matched on external AUC (+0.0034, p=0.007) — more training data helps,
in the expected direction and magnitude.

### F1 (revised) — CONFIRMED, and now significant in the opposite direction
FedAvg is **significantly worse** than the matched centralised control on external
AUC (−0.0049, p=0.048), as are FedNova and EqualW. The paper's "FedAvg attains the
best external AUC" is not merely unsupported; the reverse is significant.

### F2 (revised) — the gap advantage holds for FedProx ONLY
FedAvg's Δgen-gap is −0.0070, p=0.063 — **not significant** even at 10 seeds.
Only FedProx clears the bar (−0.0117, p=7e−4). The claim "federation reduces the
generalisation gap" is therefore defensible for FedProx and not for FedAvg.
XGBoost is significantly *worse* (+0.0258, p=7e−13).

### F3 (revised) — the fairness advantage is the strongest surviving result
Every federated variant significantly narrows the elderly–young gap
(p between 3.2e−4 and 1.8e−6); neither centralised control nor XGBoost does.
This survives the composition-matched control and 10 seeds.

### F5 (NEW, T1.5) — the subgroup decomposition separates FedProx from the rest

| model | Δ young | Δ mid | Δ elderly |
|---|---|---|---|
| **FedProx** | **−0.0134 p=9e−10** | +0.0001 p=.99 | **+0.0006 [−0.0045, +0.0056] p=.83** |
| FedAvg | −0.0214 p=4e−14 | −0.0100 p=.013 | −0.0059 [−0.0143, +0.0024] p=.16 |
| FedNova | −0.0213 p=2e−14 | −0.0106 p=.007 | −0.0065 p=.14 |
| FedAvg-EqualW | −0.0244 p=1e−17 | −0.0143 p=5e−4 | −0.0086 p=.041 |

FedAvg/FedNova/EqualW degrade **all three** age bands — the disparity closes by
levelling down everywhere (ChatGPT's Case D, the weakest reading).
FedProx degrades the young band only and leaves the elderly band statistically
unchanged. Still levelling down, but confined to the advantaged group, and with
no cost to the under-served group or to overall external AUC.

### F4 (confirmed at 10 seeds) — client weighting is still not the mechanism
EqualW's fairness Δ (−0.0158) is indistinguishable from FedAvg's (−0.0154)
despite genuinely changing the aggregation weights. Combined with the algebraic
point that FedAvg's effective per-sample weight is 1/N — identical to pooled
centralised — `Centralised-matched` already *is* the weighting-matched control.
Neither composition nor weighting explains the effect.

### T0.3b — `ece_cis.json` regenerated; three further defects found and fixed
Beyond the known Platt-transform bug, this file disagreed with
`17_fedavg_calibration.py` for three more reasons, each sufficient on its own:
2. ECE was evaluated on all 1,282,897 rows including the 20% the calibrators were
   fitted on, instead of the held-out 80%;
3. the bin convention was `(lo, hi]` first-bin-closed; `ece_wilson()` uses
   `[lo, hi)` last-bin-closed;
4. **the isotonic CI was not a bootstrap** — it was Platt's bootstrap width
   recentred on the isotonic point estimate. Isotonic is now refitted and
   bootstrapped like every other method.
Also deleted `ece_from_preds()`, dead code that referenced an undefined `i`.
All four methods now reproduce `17_fedavg_calibration.py` exactly
(`matches_17: true`), and the paper's quoted 0.276 / 0.007 / 0.001 now carry real
CIs: Uncalibrated 0.2762 [0.2754, 0.2769], Platt 0.0072 [0.0066, 0.0079],
Isotonic 0.0010 [0.0007, 0.0018], Temperature 0.2687 [0.2680, 0.2694].

### Revised paper story — now defensible, but it is a DIFFERENT paper
> Federated optimisation acts as a regulariser rather than a source of predictive
> gain. Against a composition-matched centralised control, FedProx matches
> external discrimination exactly (Δ=+0.0000, p=0.98) while significantly
> reducing both the internal–external generalisation gap (−0.0117, p=7e−4) and the
> elderly–young AUC disparity (−0.0140, p=1.8e−6). Neither effect is attributable
> to training-set composition or to client-level weighting, which points to the
> local-epoch/averaging dynamics themselves. The disparity narrows because
> performance falls in the better-served younger band while the elderly band is
> unchanged — a levelling-down effect that should be reported as such, not as a
> benefit to older patients. FedAvg, the strategy the current manuscript is built
> on, does not support these claims: it is significantly *worse* than the
> centralised control on external AUC and its gap reduction is not significant.

**Action required: the manuscript is built on FedAvg and must be rebuilt on
FedProx.** This is a headline-model change, not an edit.

---

## 2026-08-11 (cont.) — two implementation defects invalidate the numbers above

### D10 — [BLOCKER] BatchNorm buffers were never aggregated
`DiabetesNet` uses `nn.BatchNorm1d`. Its `weight`/`bias` are in
`model.parameters()` and so ride along with every aggregation, but
`running_mean`/`running_var` are **buffers** — not in `parameters()`, therefore
never sent, never aggregated, never restored. `params_to_model()` built a fresh
`DiabetesNet` (running_mean=0, running_var=1) and overwrote only the parameters,
so every federated model was scored in `eval()` with **untrained BN statistics**,
while `train_centralised()` returned the real module with its learned ones.

Measured at seed 42 (`results/bn_buffer_check.json`, via `36_bn_buffer_check.py`):

| strategy | external, default BN | external, aggregated BN | Δ |
|---|---|---|---|
| FedAvg | 0.74063 | 0.74573 | **+0.00509** |
| FedProx | 0.74426 | 0.74506 | +0.00080 |

**The handicap (+0.0051) is the same size as the entire F1 effect (−0.0049) and
in the opposite direction.** F1 — "FedAvg is significantly worse than the matched
centralised control" — is therefore probably an artefact and must be recomputed.
Internal AUC gains almost the same amount (+0.00479), so the generalisation-gap
result is largely insulated; the fairness numbers are not and must be redone.

This affects the legacy pipeline too: `03_federated_simulation.py` and
`07_external_validation.py` share the defect, so **every federated number
currently in the manuscript is affected**, not only the multi-seed ones.

Fixed in `fl_core.py`: buffers are now returned by `local_train`, pooled by
`aggregate_buffers()` with client-size weighting, and restored before scoring.
`running_var` pools with the exact identity Var = E[x²] − (E[x])², not a weighted
mean of client variances, because the clients are deliberately non-IID and the
between-client mean shift is not negligible.

### D11 — [BLOCKER] the SCAFFOLD "negative result" is a training-budget artefact
`08_scaffold_baseline.py:148` runs `while step < K` with `K = NN_LOCAL_EPOCHS = 5`
and increments `step` once per **batch**. So SCAFFOLD performed 5 mini-batch steps
per round where every other strategy performed 5 full epochs (~90 batches) —
about 1/18th the local budget. It also used SGD at `ETA_LOCAL = NN_LR = 1e-3`,
an AdamW-tuned learning rate.

The manuscript reports this as science, at `main.tex:139` (table row
`SCAFFOLD (SGD) 0.642 / 0.665`) and `main.tex:187`, a paragraph headed
*"SCAFFOLD: an instructive negative result."*

LR sweep on held-out node validation AUC (`37_scaffold_lr_sweep.py`, BRFSS never
touched), seed 42, with the local budget matched to the other strategies:

| eta_local | internal val AUC |
|---|---|
| 0.001 (the old value) | 0.7413 |
| 0.01 | 0.7805 |
| 0.03 | 0.7872 |
| 0.1 | 0.7879 |
| 0.3 | 0.7913 |
| **1.0 (selected)** | **0.7970** |
| 3.0 | 0.5000 (diverged) |

Interior optimum at 1.0. Properly configured, SCAFFOLD reaches internal AUC
0.797 against FedAvg's ~0.784 — it does **not** underperform. The paragraph is
false as written and must be removed or rewritten.

`run_scaffold()` in `fl_core.py` is a reimplementation with the local budget
matched. It keeps SGD locally, because the control-variate correction is derived
for SGD and applying it to AdamW's rescaled step has no such guarantee; that
optimiser difference against the four AdamW strategies must be stated when
reported.

### Consequence
All Δ statistics in the previous entry were computed on pre-D10 predictions and
are superseded. The prediction cache must be regenerated and
`35_delong_deltas.py` rerun before anything is written into the manuscript.

---

## 2026-08-11 (cont.) — D12: the race feature was miscoded across the train/external boundary

### D12 — [BLOCKER] three independent defects in `RIDRETH3`
1. **Incompatible code sets.** Training uses NHANES `RIDRETH3` ∈ {1,2,3,4,6,7}
   (1=Mexican American, 2=Other Hispanic, 3=NH White, 4=NH Black, 6=NH Asian,
   7=Other/Multi). External uses BRFSS `_RACE` ∈ {1..9} (1=NH White, 2=NH Black,
   3=AI/AN, 4=Asian, 5=NHPI, 6=Other, 7=Multiracial, 8=Hispanic, 9=Refused).
   BRFSS "1" — 618,605 rows, 48% of the external set — was read by the model as
   *Mexican American*. Verified directly from both processed CSVs.
2. **34.7% missing, mode-imputed.** 445,133 of 1,282,897 BRFSS rows have race
   missing; `load_brfss_clean` filled them with the mode, assigning a third of
   the external set to a single category.
3. **Ordinal encoding of a nominal variable.** Fed as a raw scaled scalar, so the
   model treated 1 < 2 < 3 < 4 < 6 < 7 as ordered.

Same defect class as the sex 0/1-vs-1/2 mismatch fixed earlier (D1b); race was
missed because `load_brfss_clean` only remapped sex. **Every external number in
the manuscript is computed with this corrupted column.**

### Resolution (user decision): drop race as an input, stratify evaluation by it
Race is removed from the feature set (8 → 7 features) and retained only as an
evaluation grouping. This avoids having to defend a BRFSS→NHANES crosswalk for a
model input, matches current practice for race-free clinical prediction models,
and gives a cleaner fairness audit — group membership is no longer entangled
with a corrupted model input.

Implementation:
- `config_paths.FEATURE_COLS_NORACE` (7 cols) + `GLOBAL_SCALER_NORACE_PATH`
- `39_fit_norace_scaler.py` refits the global scaler on the same NHANES train
  split. Asserts the retained columns' means/scales match the 8-feature scaler
  exactly, so dropping race perturbs nothing else. **Passed.**
- `load_brfss_clean(feature_cols=...)` now imputes only columns the model
  consumes, and preserves pre-imputation race in `RIDRETH3_RAW` so the missing
  third stays its own stratum instead of vanishing into the mode.
- `fl_core.new_net()` builds `DiabetesNet(input_dim=7)`.
- `35_delong_deltas.py` adds race strata (White/Black/Hispanic/Asian/Other,
  plus Unknown reported but excluded from the disparity metric) and a
  White−Black AUC gap computed exactly like the age gap.

**Caveat to verify:** the BRFSS `_RACE` code→label mapping in
`data_utils.BRFSS_RACE_GROUPS` is taken from the codebook and is consistent with
the observed prevalences (Black 19.5%, AI/AN 19.9%, White 12.4%, Asian 9.6%),
but should be confirmed against the codebook for the exact survey years before
it reaches the manuscript. A mislabel would rename a stratum, not corrupt a
prediction.

### Seed-42 smoke test (race-free + BN fix) — the fairness result may reverse

| model | ext AUC | fairness gap |
|---|---|---|
| Centralised-matched | 0.7569 | 0.0656 |
| Centralised-full | 0.7586 | 0.0632 |
| FedAvg | 0.7506 | **0.0742** |
| FedProx | 0.7479 | **0.0824** |
| SCAFFOLD | 0.7531 | 0.0656 |
| XGBoost | 0.7328 | 0.0639 |

At this one seed the federated fairness gap is **larger** than the centralised
control's — the reverse of F3. If that holds across 10 seeds, the fairness
advantage was an artefact of the miscoded race feature and/or the BN bug, and F3
must be withdrawn along with F1. **Do not write anything until the 10-seed run
lands.** Also note external AUCs rose for every model once the corrupted column
was removed (Centralised-matched 0.7524 → 0.7569).

---

## 2026-08-11 (final) — corrected 10-seed results. Three claims die, one strong new one appears.

Pipeline: race-free 7-feature models, BN buffers aggregated, SCAFFOLD at matched
local budget, seeds 42–51, scored on 1,282,897 BRFSS rows.
`results/multiseed_all_models.json`, `results/delong_deltas.json`.

### Absolute values (10-seed means)

| model | ext AUC | gen gap | age gap | White | Black | White−Black |
|---|---|---|---|---|---|---|
| XGBoost | 0.7330 | 0.0356 | 0.0636 | 0.7293 | 0.7335 | −0.0041 |
| Centralised-full | 0.7576 | 0.0286 | 0.0650 | 0.7574 | 0.7511 | +0.0063 |
| Centralised-matched | 0.7543 | 0.0344 | 0.0672 | 0.7537 | 0.7462 | +0.0075 |
| FedAvg | 0.7530 | 0.0298 | 0.0654 | 0.7533 | 0.7528 | **+0.0005** |
| FedProx | 0.7522 | 0.0298 | 0.0717 | 0.7521 | 0.7534 | −0.0013 |
| FedNova | 0.7528 | 0.0300 | 0.0657 | 0.7532 | 0.7528 | +0.0004 |
| FedAvg-EqualW | 0.7531 | 0.0301 | 0.0656 | 0.7534 | 0.7522 | +0.0012 |
| SCAFFOLD | 0.7541 | 0.0293 | 0.0602 | 0.7547 | 0.7529 | +0.0019 |

### Verdicts on every prior finding

**F1 — "FedAvg attains the best external AUC": FALSE, but the corrected reading
is favourable.** FedAvg −0.0013 [−0.0030, +0.0005] p=0.16 vs the matched control;
SCAFFOLD −0.0001 p=0.91. Federation is statistically indistinguishable from a
composition-matched centralised model. Centralised-full is genuinely best
(+0.0033, p=3.8e−6). The manuscript's 0.757 is dead; "federation costs nothing
in discrimination" is supportable and is the honest replacement.

**F2 — the generalisation-gap advantage: DEAD.** No model differs significantly
from the matched control (FedAvg −0.0046, p=0.25; sign 6−/4+). The earlier
p=7e−4 for FedProx was an artefact of the BN bug and the miscoded race column.
The manuscript's "40% smaller gap" claim must be removed entirely.

**F3 — the age-fairness advantage: DEAD.** FedAvg −0.0019, p=0.34. FedProx is
significantly **worse** (+0.0045, p=0.023). Only SCAFFOLD narrows it (−0.0070,
p=1.2e−5). The "21.7% reduction in the elderly–young gap" is not reproducible.

**F4 — client weighting is not the mechanism: CONFIRMED.** FedAvg-EqualW tracks
FedAvg within 0.001 on every metric, at 10 seeds.

**F5 — "SCAFFOLD: an instructive negative result": FALSE.** Given a matched local
budget and a properly selected eta_local, SCAFFOLD ties the centralised control
on external AUC (−0.0001, p=0.91), has the smallest federated generalisation gap
(0.0293), and the best age-fairness gap of any model (0.0602). The manuscript's
paragraph and table row are artefacts of D11 and must be deleted.

### F6 — NEW, and the strongest result in the study: federation levels UP by race

Δ vs Centralised-matched, all five federated strategies, 10/10 seeds:

| stratum | n | FedAvg Δ | p |
|---|---|---|---|
| White | 617,782 | −0.0004 | 0.73 (ns) |
| **Black** | 62,770 | **+0.0066** | **6.6e−11** |
| **Hispanic** | 74,496 | **+0.0039** | **1.7e−6** |
| **Asian** | 21,467 | **+0.0087** | **1.7e−7** |
| Other | 42,779 | −0.0021 | 0.059 (ns) |

White−Black gap: **0.0075 → 0.0005**, Δ = −0.0070 [−0.0092, −0.0048],
p=3.5e−10. Every federated strategy: FedProx −0.0088 (p=2.7e−15), FedNova
−0.0071, EqualW −0.0064, SCAFFOLD −0.0057. All 10/10 seeds.

This is levelling **up**, not down: minority-stratum AUC rises while the majority
stratum is statistically unchanged. Contrast XGBoost, which also narrows the gap
(−0.0117) but by degrading everyone (White −0.0244, Black −0.0127) — the
levelling-down pattern the age analysis previously showed.

Two controls rule out the obvious confounds:
- **Not data quantity.** Centralised-full improves every stratum roughly
  uniformly (White +0.0037, Black +0.0050) so its race gap does not close
  (−0.0012, p=0.11). More data helps everyone; federation helps the minority.
- **Not client weighting.** EqualW ≈ FedAvg (−0.0064 vs −0.0070), and FedAvg's
  effective per-sample weight is already 1/N, identical to pooled training.

Remaining mechanism: the local-epoch + periodic-averaging dynamics. Each node
trains to its own local optimum before averaging, so the majority stratum does
not dominate the shared representation the way it does under pooled SGD.

Note this was a pre-planned analysis (T2.1, from D4 — race was a model input but
never audited), specified before any of these results were seen. It is not a
post-hoc substitute for the age finding, and the age finding's death is reported
above rather than buried.

### Revised paper story (supported by the corrected numbers)
> Auditing five federated strategies against a composition-matched centralised
> control on 1.28M external respondents, federation costs nothing in overall
> discrimination and closes the White–Black AUC gap from 0.0075 to 0.0005 by
> raising minority-subgroup performance, not by degrading the majority. The
> effect holds for every strategy across 10 seeds and is attributable to neither
> training-set composition nor client weighting. Three claims from the prior
> version of this work — a federated advantage in overall AUC, in the
> generalisation gap, and in the elderly–young gap — do not survive correction of
> four pipeline defects, and are reported as non-replications.

### Next
- [x] 10-seed race-free run + rerun 35
- [ ] Re-examine F1/F2/F3/F5 against the corrected numbers
- [ ] Rewrite manuscript: family framing + FedProx headline (user decision), only
      if the corrected results still support it
- [ ] `08_scaffold_baseline.py` still contains the D11 budget bug; either fix or
      delete it so the repo has one SCAFFOLD implementation (`fl_core.run_scaffold`)
- [ ] T2.1 — race-stratified external fairness (RIDRETH3 now cached)
- [ ] T2.2 — SCAFFOLD multi-seeded through fl_core for a complete strategy table
- [ ] T2.6 — AUPRC (young prevalence 2.4% vs elderly 20.2%; AUC alone is thin)
- [ ] T2.5 — NOT POSSIBLE on BRFSS: the processed CSV has no survey-weight/PSU/strata
      columns. NHANES-only, or drop the item.
