"""
write_supplementary_v2.py
=========================
Generates FL_Diabetes_Supplementary_v6.docx for COBM (Elsevier) submission.

Changes from write_supplementary.py (v1):
  S1. Cover page: JBI → COBM, Tables S1–S3 → S1–S5
  S2. Table S2: DP ε levels extended (10, 20, 50, 100 added via config_paths)
  S3. Table S3: Node A age range confirmed correct ("18–44 or minority")
  S4. New Table S4: Pairwise DeLong matrix (all 3 FL strategy comparisons)
  S5. New Table S5: FedProx post-hoc calibration (moved from main Table 2)

Usage: python write_supplementary_v2.py
"""

import os, sys, json
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NN_HIDDEN_DIMS, NN_DROPOUT, NN_LR, NN_WEIGHT_DECAY,
    NN_BATCH_SIZE, NN_LOCAL_EPOCHS, FL_NUM_ROUNDS, FL_NUM_CLIENTS,
    FEDPROX_MU, NODE_LOCAL_EPOCHS_FEDNOVA, DP_TARGET_DELTA,
    DP_MAX_GRAD_NORM, DP_EPSILON_LEVELS, SEED, CI_ALPHA,
)

ROOT    = os.path.join(os.path.dirname(__file__), '..')
RESDIR  = os.path.join(os.path.dirname(__file__), 'results')
OUTFILE = os.path.join(ROOT, 'FL_Diabetes_Supplementary_v6.docx')


# ── Load experimental results ──────────────────────────────────────────────────
def load_json(fname):
    with open(os.path.join(RESDIR, fname)) as f:
        return json.load(f)

pdel = load_json('pairwise_delong_all.json')
fcal_prox = load_json('calibration_results.json')   # FedProx calibration (09_calibration)
fcal_avg  = load_json('fedavg_calibration_results.json')

print(f"  FedProx calibration: ECE_raw={fcal_prox['uncalibrated']['ece']:.4f}  "
      f"ECE_platt={fcal_prox['platt']['ece']:.4f}  ECE_iso={fcal_prox['isotonic']['ece']:.4f}")
print(f"  FedAvg  calibration: ECE_raw={fcal_avg['uncalibrated']['ece']:.4f}  "
      f"ECE_platt={fcal_avg['platt']['ece']:.4f}  ECE_iso={fcal_avg['isotonic']['ece']:.4f}")
print(f"  Pairwise DeLong: {len(pdel['tests'])} comparisons loaded")


# ── helpers ───────────────────────────────────────────────────────────────────
def shade_cell(cell, fill_hex):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  fill_hex)
    tcPr.append(shd)


def make_table(doc, headers, rows, col_widths=None,
               hdr_fill='DBEAFE', alt_fill='F8FAFC'):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hrow = table.rows[0]
    for j, h in enumerate(headers):
        cell = hrow.cells[j]
        cell.text = h
        run = cell.paragraphs[0].runs[0] if cell.paragraphs[0].runs \
              else cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade_cell(cell, hdr_fill)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        if r_idx % 2 == 1:
            for cell in row.cells:
                shade_cell(cell, alt_fill)
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            run = cell.paragraphs[0].runs[0] if cell.paragraphs[0].runs \
                  else cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

    if col_widths:
        for r_idx in range(len(table.rows)):
            for c_idx, w in enumerate(col_widths):
                table.rows[r_idx].cells[c_idx].width = Inches(w)

    return table


def add_table_title(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)
    return p


def add_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(10)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    return p


# ── Document setup ────────────────────────────────────────────────────────────
doc = Document()
sec = doc.sections[0]
sec.page_width  = Inches(8.5)
sec.page_height = Inches(11)
sec.left_margin = sec.right_margin = Inches(1.0)
sec.top_margin  = sec.bottom_margin = Inches(1.0)

style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)
style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE


# ── S1: Cover page (COBM, Tables S1–S5) ──────────────────────────────────────
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_p.paragraph_format.space_before = Pt(36)
title_p.paragraph_format.space_after  = Pt(12)
tr = title_p.add_run('Supplementary Material')
tr.bold = True
tr.font.size = Pt(16)

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_p.paragraph_format.space_after = Pt(8)
subr = sub_p.add_run(
    'Privacy-Preserving Federated Learning for Diabetes Risk Prediction\n'
    'Across Demographically Heterogeneous Hospital Nodes'
)
subr.font.size = Pt(12)
subr.bold = True

journal_p = doc.add_paragraph()
journal_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
journal_p.paragraph_format.space_after = Pt(36)
journalr = journal_p.add_run(
    'Computers in Biology and Medicine (COBM), Elsevier\n'
    'Supplementary Tables S1–S5\n'
    'Rajveer Singh Pall and Sameer Yadav\n'
    'Gyan Ganga Institute of Technology and Sciences, Jabalpur, India'
)
journalr.font.size = Pt(10)
journalr.italic = True

toc_p = doc.add_paragraph()
toc_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
toc_p.paragraph_format.space_before = Pt(20)
toc_r = toc_p.add_run(
    'Contents\n'
    '  Table S1 — Feature Harmonisation Mapping (NHANES ↔ BRFSS)\n'
    '  Table S2 — Full Hyperparameter Specification\n'
    '  Table S3 — Node Partition Details\n'
    '  Table S4 — Pairwise DeLong Tests Between FL Strategies (BRFSS External)\n'
    '  Table S5 — FedProx Post-Hoc Calibration (BRFSS External)'
)
toc_r.font.size = Pt(10)

doc.add_page_break()


# ════════════════════════════════════════════════════════════
# TABLE S1: Feature Harmonisation Mapping (UNCHANGED)
# ════════════════════════════════════════════════════════════
add_table_title(doc,
    'Table S1. Feature Harmonisation Mapping Between NHANES and BRFSS')
p_intro = doc.add_paragraph(
    'Eight predictor features (FEATURE_COLS) enter the DiabetesNet model; '
    'rows 1–8 below are the model inputs harmonised across NHANES 2013–2020 and '
    'BRFSS 2020–2022. Row 9 describes the diabetes outcome (prediction target). '
    'Measurement invariance risk indicates differences in variable definitions '
    'and collection protocols across surveys.'
)
p_intro.paragraph_format.space_after = Pt(6)
for run in p_intro.runs:
    run.font.size = Pt(10)

s1_headers = [
    'Feature', 'NHANES Variable', 'BRFSS Variable',
    'Mapping Notes', 'Known Measurement Differences',
]
s1_rows = [
    ['Age',
     'RIDAGEYR (continuous, years)',
     'AGEG5YR (13 age groups, midpoint used)',
     'BRFSS group midpoint used as continuous proxy',
     'Low: slight grouping error at group boundaries'],
    ['Sex',
     'RIAGENDR (1=Male, 2=Female)',
     '_SEX (1=Male, 2=Female)',
     'Direct match; same coding',
     'Low: consistent definition across surveys'],
    ['Race/Ethnicity',
     'RIDRETH3 (6 categories)',
     '_RACE (5 categories)',
     'NHANES 6-group to BRFSS 5-group alignment; Hispanic merged',
     'Medium: Hispanic sub-group definitions differ slightly'],
    ['BMI',
     'BMXBMI (measured, kg/m²)',
     '_BMI5 / 100 (self-reported)',
     'BRFSS _BMI5 divided by 100 to restore kg/m² scale',
     'Medium: self-report underestimates ~1.5 kg/m² systematically'],
    ['Smoking status',
     'SMQ020 / SMQ040 (ever / current smoker)',
     'SMOKE100 / SMOKDAY2 (100 cigarettes / current)',
     'Mapped to 3-level: never=0, former=1, current=2',
     'High: NHANES lab-verified cotinine; BRFSS self-reported; residual mismatch'],
    ['Physical activity',
     'PAQ605 / PAQ620 / PAQ635 / PAQ650 / PAQ665 (any domain)',
     'EXERANY2 (any leisure activity, binary)',
     'Any domain reporting "Yes" → active (1); else inactive (0)',
     'Medium: NHANES multi-domain minutes; BRFSS single binary flag'],
    ['Myocardial infarction history',
     'MCQ160E (ever told MI, binary)',
     'CVDINFR4 (ever told MI, binary)',
     'Direct match; both self-reported history of MI',
     'Low: both self-reported; consistent coding'],
    ['Stroke history',
     'MCQ160F (ever told stroke, binary)',
     'CVDSTRK3 (ever told stroke, binary)',
     'Direct match; both self-reported stroke history',
     'Low: both self-reported; consistent coding'],
    ['Diabetes (outcome)',
     'DIQ010 + LBXGH + LBXGLU (composite)',
     'DIABETE4 (self-reported diagnosis)',
     'NHANES: composite of self-report, HbA1c ≥6.5%, fasting glucose ≥126 mg/dL',
     'Medium: NHANES clinically confirmed; BRFSS self-reported diagnosis only'],
]
make_table(doc, s1_headers, s1_rows, col_widths=[1.0, 1.3, 1.2, 1.7, 1.8])
add_note(doc,
    'All 8 predictor features (rows 1–8) are available in both NHANES and BRFSS '
    'and enter the DiabetesNet model after StandardScaler transformation. '
    'The diabetes outcome (row 9) is the prediction target, not a predictor. '
    'Sensitivity analysis with smoking excluded showed AUC change < 0.003, '
    'confirming that the smoking measurement mismatch does not materially '
    'affect model performance.')

doc.add_page_break()


# ════════════════════════════════════════════════════════════
# S2: TABLE S2: Full Hyperparameter Specification
#     (DP ε levels extended via config_paths.py)
# ════════════════════════════════════════════════════════════
add_table_title(doc, 'Table S2. Full Hyperparameter Specification')
p_intro2 = doc.add_paragraph(
    'All hyperparameters are reported as used in the final experiments. '
    'Hyperparameters marked with * were determined by grid search; '
    'others follow established defaults or theoretical recommendations.'
)
p_intro2.paragraph_format.space_after = Pt(6)
for run in p_intro2.runs:
    run.font.size = Pt(10)

s2_headers = ['Component', 'Parameter', 'Value', 'Applies To']
s2_rows = [
    # DiabetesNet architecture
    ['DiabetesNet', 'Input dimension',
     '8 (= N_FEATURES; see FEATURE_COLS in config_paths.py)',
     'All FL and centralised models'],
    ['DiabetesNet', 'Hidden layer sizes',
     ' → '.join(str(d) for d in NN_HIDDEN_DIMS),
     'All FL and centralised models'],
    ['DiabetesNet', 'Dropout rate (layers 1–2)',
     str(NN_DROPOUT),
     'All FL and centralised models'],
    ['DiabetesNet', 'Dropout rate (layer 3)',
     '0.18',
     'All FL and centralised models'],
    ['DiabetesNet', 'Batch normalisation',
     'Yes (after each hidden layer)',
     'All FL and centralised models'],
    ['DiabetesNet', 'Output',
     'Dense(1) logit — BCEWithLogitsLoss',
     'All FL and centralised models'],
    ['DiabetesNet', 'Total parameters',
     '4,993',
     'All FL and centralised models'],

    # Optimiser
    ['Optimiser', 'Algorithm',
     'AdamW',
     'FedAvg, FedProx, FedNova, centralised NN'],
    ['Optimiser', 'Learning rate (lr)*',
     str(NN_LR) + ' (grid-searched: {1e-4, 1e-3, 5e-3})',
     'FedAvg, FedProx, FedNova, centralised NN'],
    ['Optimiser', 'Weight decay (λ)',
     str(NN_WEIGHT_DECAY),
     'FedAvg, FedProx, FedNova, centralised NN'],
    ['Optimiser', 'Scheduler',
     'CosineAnnealingLR (T_max = FL_NUM_ROUNDS)',
     'FedAvg, FedProx, FedNova, centralised NN'],
    ['Optimiser', 'SCAFFOLD algorithm',
     'SGD (lr = 1e-3, η_l = 1e-3)',
     'SCAFFOLD only (required for theoretical guarantees)'],

    # Training
    ['Training', 'Batch size',
     str(NN_BATCH_SIZE),
     'All local node training'],
    ['Training', 'Local epochs per round',
     str(NN_LOCAL_EPOCHS),
     'FedAvg, FedProx, SCAFFOLD'],
    ['Training', 'Mixed precision (AMP)',
     'Yes (CUDA) / No (CPU)',
     'All models'],
    ['Training', 'Random seed',
     str(SEED),
     'All experiments'],

    # Federated learning
    ['Federated', 'Communication rounds',
     str(FL_NUM_ROUNDS),
     'All FL strategies'],
    ['Federated', 'Number of clients (nodes)',
     str(FL_NUM_CLIENTS),
     'All FL strategies'],
    ['Federated', 'FedProx proximal μ*',
     str(FEDPROX_MU) + ' (grid-searched: {0.01, 0.1, 0.5})',
     'FedProx only'],

    # FedNova
    ['FedNova', 'Node A local epochs (τ_A)',
     str(NODE_LOCAL_EPOCHS_FEDNOVA[0]) + ' (Young Urban: low shift)',
     'FedNova only'],
    ['FedNova', 'Node B local epochs (τ_B)',
     str(NODE_LOCAL_EPOCHS_FEDNOVA[1]) + ' (Elderly Rural: high shift)',
     'FedNova only'],
    ['FedNova', 'Node C local epochs (τ_C)',
     str(NODE_LOCAL_EPOCHS_FEDNOVA[2]) + ' (Mixed Metro: intermediate)',
     'FedNova only'],

    # SCAFFOLD
    ['SCAFFOLD', 'Option',
     'Option II (Karimireddy et al. ICML 2020)',
     'SCAFFOLD only'],
    ['SCAFFOLD', 'Control variate steps K',
     '5 (= NN_LOCAL_EPOCHS)',
     'SCAFFOLD only'],
    ['SCAFFOLD', 'Local learning rate η_l',
     '1e-3',
     'SCAFFOLD only'],
    ['SCAFFOLD', 'Global learning rate η_g',
     '1.0 (Option II standard)',
     'SCAFFOLD only'],

    # S2: Differential Privacy — extended ε set
    ['Diff. Privacy', 'Framework',
     'Opacus ≥ 1.4 (DP-SGD)',
     'DP experiments only'],
    ['Diff. Privacy', 'Accountant',
     'RDP (Rényi DP)',
     'DP experiments only'],
    ['Diff. Privacy', 'Gradient clipping norm C',
     str(DP_MAX_GRAD_NORM),
     'DP experiments only'],
    ['Diff. Privacy', 'DP batch size',
     '512',
     'DP experiments only'],
    ['Diff. Privacy', 'δ (failure probability)',
     str(DP_TARGET_DELTA),
     'DP experiments only'],
    ['Diff. Privacy', 'ε levels tested (extended sweep)',
     ', '.join(str(e) if e != float('inf') else '∞' for e in DP_EPSILON_LEVELS),
     'DP experiments only (0.5–5 collapse; ≥10 viable)'],

    # Statistical
    ['Statistics', 'Bootstrap resamples (internal CI)',
     '2,000',
     'AUC confidence intervals'],
    ['Statistics', 'Bootstrap method',
     'Stratified percentile',
     'AUC confidence intervals'],
    ['Statistics', 'External CI method',
     'DeLong structural components estimator',
     'BRFSS external AUC'],
    ['Statistics', 'Calibration bins',
     '10 equal-width probability bins',
     'ECE computation'],
    ['Statistics', 'Calibration split (BRFSS)',
     '20% calibration / 80% test',
     'Post-hoc calibration (FedAvg and FedProx)'],
    ['Statistics', 'Pairwise comparisons',
     '3 (FedAvg/FedProx, FedAvg/FedNova, FedProx/FedNova)',
     'Bonferroni-corrected α = 0.0167'],
]

make_table(doc, s2_headers, s2_rows, col_widths=[1.1, 1.7, 2.4, 1.8])
add_note(doc,
    '* Grid-searched hyperparameters. DiabetesNet dropout 0.18 in layer 3 '
    'was set to reduce regularisation in the narrower final hidden layer. '
    'SCAFFOLD uses SGD rather than AdamW to satisfy the theoretical convergence '
    'guarantees of Karimireddy et al. (ICML 2020); the resulting lower AUC '
    '(0.642 vs. 0.788 for FedAvg) reflects the combined effect of the '
    'algorithm and the optimiser choice (Section 5.2). '
    'DP extended sweep adds ε ∈ {10, 20, 50, 100}: model recovers at ε ≥ 10 '
    '(AUC = 0.769); collapse observed at ε ≤ 5 (per-node n ≈ 3,000–4,500).')

doc.add_page_break()


# ════════════════════════════════════════════════════════════
# S3: TABLE S3: Node Partition Details
#     (Node A already correct: "18–44 or minority")
# ════════════════════════════════════════════════════════════
add_table_title(doc, 'Table S3. Node Partition Details')
p_intro3 = doc.add_paragraph(
    'NHANES 2013–2020 records (n = 15,650 after QC exclusions) were partitioned '
    'into three federated nodes by demographic stratification. Node sizes reflect '
    'population-proportional sampling; the three nodes together comprise '
    'n = 11,906 records. The remaining 3,744 records are retained in the '
    'centralised dataset for baseline evaluation but are not assigned to any '
    'federated node. Training and validation splits (80/20 stratified by diabetes '
    'prevalence) were applied independently within each node.'
)
p_intro3.paragraph_format.space_after = Pt(6)
for run in p_intro3.runs:
    run.font.size = Pt(10)

s3_headers = [
    'Node', 'Population Description', 'Age Range',
    'Total n', 'Training n (80%)', 'Validation n (20%)',
    'Diabetes Prevalence (%)',
]
s3_rows = [
    ['Node A',
     'Young Urban — community health clinic\n(young adults + minority racial/ethnic groups)',
     '18–44 or minority',
     '4,500', '3,600', '900',
     '~8% (lower: younger cohort)'],
    ['Node B',
     'Elderly Rural — critical access hospital\n(elderly White/Mexican-American; highest shift)',
     '≥55 (White or Mexican-American)',
     '3,406', '2,725', '681',
     '~28% (higher: elderly cohort)'],
    ['Node C',
     'Mixed Metropolitan — academic medical centre\n(remaining mixed age/geography)',
     'Mixed (remainder)',
     '4,000', '3,200', '800',
     '~15% (population average)'],
    ['All nodes combined',
     'Total federated training data',
     '18+',
     '11,906', '9,525', '2,381',
     '~16% (weighted average)'],
    ['Centralised (unpartitioned)',
     'Full NHANES after QC; used for centralised baseline',
     '18+',
     '15,650', '12,520 (80%)', '3,130 (20%)',
     '~16%'],
    ['BRFSS (external)',
     'External validation only; not partitioned into nodes',
     '18+',
     '1,282,897', 'N/A', 'N/A',
     '~13.3%'],
]

make_table(doc, s3_headers, s3_rows,
           col_widths=[0.9, 1.9, 1.1, 0.7, 0.9, 0.9, 1.2])
add_note(doc,
    'Node A sampling criterion: RIDAGEYR < 45 OR minority race (RIDRETH3 ∈ {2, 4}). '
    'Node B criterion: RIDAGEYR ≥ 55 AND White or Mexican-American (RIDRETH3 ∈ {1, 3}). '
    'Node C: all remaining records after A and B assignment. '
    'Prevalence values are approximate (from actual data partitions). '
    'Training n for Node B: 2,725 from stratified 80/20 split of 3,406 records '
    '(sklearn train_test_split, SEED=42). '
    'BRFSS diabetes prevalence from external_validation.json (13.32%).')

doc.add_page_break()


# ════════════════════════════════════════════════════════════
# S4: NEW TABLE S4: Pairwise DeLong Tests
# ════════════════════════════════════════════════════════════
add_table_title(doc,
    'Table S4. Pairwise DeLong Tests Between FL Strategies (BRFSS External, n = 1,282,897)')
p_intro4 = doc.add_paragraph(
    f'All three pairwise comparisons between federated learning strategies were conducted '
    f'on BRFSS 2020–2022 (n = {pdel["n"]:,}) using the paired DeLong structural '
    f'components estimator (two-sided). Bonferroni correction applied for three '
    f'simultaneous comparisons (α_Bonferroni = {pdel["bonferroni_alpha"]:.4f}). '
    f'p-values are reported as < 0.001 where exact values fall below floating-point precision.'
)
p_intro4.paragraph_format.space_after = Pt(6)
for run in p_intro4.runs:
    run.font.size = Pt(10)

t4 = pdel['tests']
fa_fp = t4['FedAvg_vs_FedProx']
fa_fn = t4['FedAvg_vs_FedNova']
fp_fn = t4['FedProx_vs_FedNova']

def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'

def fmt_sig(t):
    return 'Yes' if t['significant_bonferroni'] else 'No'

s4_headers = [
    'Comparison',
    'AUC (Model 1)',
    'AUC (Model 2)',
    'ΔAUC (M1−M2)',
    'z-statistic',
    'p-value',
    'Bonferroni-significant?',
]
s4_rows = [
    ['FedAvg vs. FedProx',
     f"{fa_fp['auc1']:.4f}",
     f"{fa_fp['auc2']:.4f}",
     f"{fa_fp['delta']:+.4f}",
     f"{fa_fp['z']:.2f}",
     fmt_p(fa_fp['p_value']),
     fmt_sig(fa_fp)],
    ['FedAvg vs. FedNova',
     f"{fa_fn['auc1']:.4f}",
     f"{fa_fn['auc2']:.4f}",
     f"{fa_fn['delta']:+.4f}",
     f"{fa_fn['z']:.2f}",
     fmt_p(fa_fn['p_value']),
     fmt_sig(fa_fn)],
    ['FedProx vs. FedNova',
     f"{fp_fn['auc1']:.4f}",
     f"{fp_fn['auc2']:.4f}",
     f"{fp_fn['delta']:+.4f}",
     f"{fp_fn['z']:.2f}",
     fmt_p(fp_fn['p_value']),
     fmt_sig(fp_fn)],
]
make_table(doc, s4_headers, s4_rows,
           col_widths=[1.5, 1.0, 1.0, 1.0, 1.0, 0.9, 1.1])
add_note(doc,
    f'Method: Paired DeLong two-sided test. Dataset: BRFSS 2020–2022 (n = {pdel["n"]:,}). '
    f'Bonferroni-corrected significance threshold: α = {pdel["bonferroni_alpha"]:.4f} '
    f'(= 0.05 / {pdel["n_comparisons"]}). '
    'All three comparisons are statistically significant after correction, indicating '
    'that FedAvg, FedProx, and FedNova produce distinguishable discrimination '
    'performance on this large external dataset despite small absolute AUC differences. '
    'All ΔAUC values are positive for FedAvg comparisons, confirming FedAvg as '
    'the best-performing federated strategy.')

doc.add_page_break()


# ════════════════════════════════════════════════════════════
# S5: NEW TABLE S5: FedProx Post-Hoc Calibration
# ════════════════════════════════════════════════════════════
add_table_title(doc,
    'Table S5. FedProx Post-Hoc Calibration on BRFSS External Validation')
p_intro5 = doc.add_paragraph(
    f'Post-hoc probability calibration was applied to the FedProx model on the '
    f'BRFSS 2020–2022 external validation set (n = {fcal_prox["n_cal"] + fcal_prox["n_test"]:,}). '
    f'A 20/80 calibration/test split (rng seed = {SEED}) was used: '
    f'{fcal_prox["n_cal"]:,} calibration samples, {fcal_prox["n_test"]:,} test samples. '
    f'Three calibration methods are compared. ECE = Expected Calibration Error (10 equal-width bins). '
    f'AUC is preserved (calibration is rank-preserving). '
    f'Analogous results for FedAvg appear in the main text Table 2.'
)
p_intro5.paragraph_format.space_after = Pt(6)
for run in p_intro5.runs:
    run.font.size = Pt(10)

prox_ece_raw   = fcal_prox['uncalibrated']['ece']
prox_auc_raw   = fcal_prox['uncalibrated']['auc']
prox_ece_platt = fcal_prox['platt']['ece']
prox_auc_platt = fcal_prox['platt']['auc']
prox_ece_iso   = fcal_prox['isotonic']['ece']
prox_auc_iso   = fcal_prox['isotonic']['auc']
prox_ece_temp  = fcal_prox['temperature']['ece']
prox_auc_temp  = fcal_prox['temperature']['auc']
prox_T_opt     = fcal_prox['temperature']['T_optimal']

s5_headers = [
    'Calibration Method',
    'ECE (before)',
    'ECE (after)',
    'ECE Reduction',
    'AUC (test)',
    'Notes',
]

def pct_reduction(before, after):
    return f"{100*(before - after)/before:.1f}%"

s5_rows = [
    ['Uncalibrated',
     f'{prox_ece_raw:.4f}',
     '—',
     '—',
     f'{prox_auc_raw:.4f}',
     'Baseline; model overconfident'],
    ['Platt scaling',
     f'{prox_ece_raw:.4f}',
     f'{prox_ece_platt:.4f}',
     pct_reduction(prox_ece_raw, prox_ece_platt),
     f'{prox_auc_platt:.4f}',
     f'Logistic fit on calibration set; '
     f'coef={fcal_prox["platt"]["coef"]:.2f}, '
     f'intercept={fcal_prox["platt"]["intercept"]:.2f}'],
    ['Isotonic regression',
     f'{prox_ece_raw:.4f}',
     f'{prox_ece_iso:.4f}',
     pct_reduction(prox_ece_raw, prox_ece_iso),
     f'{prox_auc_iso:.4f}',
     'Non-parametric; best ECE; risk of overfitting small cal sets'],
    [f'Temperature scaling (T={prox_T_opt:.2f})',
     f'{prox_ece_raw:.4f}',
     f'{prox_ece_temp:.4f}',
     pct_reduction(prox_ece_raw, prox_ece_temp),
     f'{prox_auc_temp:.4f}',
     'Single-param rescaling; least reduction for FedProx'],
]
make_table(doc, s5_headers, s5_rows,
           col_widths=[1.7, 0.9, 0.9, 1.1, 0.9, 2.1])

# Subgroup ECE if available
sg = fcal_prox.get('subgroup_internal', {})
if sg:
    young_ece_raw   = sg.get('Young (18-39)', {}).get('ece_raw', None)
    young_ece_platt = sg.get('Young (18-39)', {}).get('ece_platt', None)
    eld_ece_raw     = sg.get('Elderly (>=60)', {}).get('ece_raw', None)
    eld_ece_platt   = sg.get('Elderly (>=60)', {}).get('ece_platt', None)
    sg_note = (
        f'Subgroup calibration (internal NHANES validation set): '
        f'Young (18–39, n={sg["Young (18-39)"]["n"]:,}): '
        f'ECE before={young_ece_raw:.3f}, after Platt={young_ece_platt:.3f}. '
        f'Elderly (≥60, n={sg["Elderly (>=60)"]["n"]:,}): '
        f'ECE before={eld_ece_raw:.3f}, after Platt={eld_ece_platt:.3f}. '
        f'Platt scaling disproportionately improves calibration in the elderly subgroup '
        f'(ECE reduction: {pct_reduction(eld_ece_raw, eld_ece_platt)} vs. '
        f'{pct_reduction(young_ece_raw, young_ece_platt)} in young adults), '
        f'consistent with systematic overconfidence for elderly patients '
        f'with high diabetes prevalence. Note: subgroup ECE computed on '
        f'internal NHANES validation set (subgroup n too small in BRFSS 20% calibration split).'
    )
    add_note(doc, sg_note)
else:
    add_note(doc,
        'ECE computed using 10 equal-width probability bins with Wilson 95% CI per bin. '
        'AUC is preserved under all calibration methods because they are monotone transformations '
        'of the original scores. Isotonic regression achieves the lowest ECE at the cost of '
        'potential overfitting on small calibration sets. '
        'For analogous FedAvg calibration results, see main manuscript Table 2.')


# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(OUTFILE)
print(f'\n  Saved: {OUTFILE}')
size_kb = os.path.getsize(OUTFILE) // 1024
print(f'  Size : {size_kb} KB')

# ── Verify ────────────────────────────────────────────────────────────────────
from docx import Document as DocxDoc
d = DocxDoc(OUTFILE)
n_tables = len(d.tables)
print(f'\n  Tables in output DOCX: {n_tables}')
assert n_tables == 5, f'Expected 5 tables, found {n_tables}'
print('  Verification PASSED: exactly 5 tables (S1–S5) present')
print('\n  Supplementary material v2 complete.')
