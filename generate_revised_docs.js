"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  ExternalHyperlink, TabStopType, TabStopPosition, LineRuleType, ImageRun,
} = require("docx");
const fs = require("fs");
const path = require("path");

// ── Figure paths & display dimensions (pt at 300 DPI → fit A4 content width 451pt) ──
const FIGS_DIR = "D:/Projects/diabetes_prediction_project/federated/results/figures";

// [widthPt, heightPt] scaled to fit 451pt content width
const FIG_DIMS = {
  "fig1_architecture.png":    [451, 245],
  "fig2_convergence.png":     [451, 267],
  "fig3_roc_curves.png":      [451, 211],
  "fig4_fairness.png":        [451, 190],
  "fig5_dp_tradeoff.png":     [451, 267],
  "fig6_calibration.png":     [451, 374],
  "fig7_generalisation_gap.png": [451, 250],
  "fig8_summary_comparison.png": [451, 214],
};

function figImage(n, filename, caption) {
  const [w, h] = FIG_DIMS[filename] || [451, 250];
  const imgData = fs.readFileSync(path.join(FIGS_DIR, filename));
  return [
    new Paragraph({
      children: [new ImageRun({ data: imgData, transformation: { width: w, height: h }, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 60, line: 240, lineRule: LineRuleType.AUTO },
    }),
    new Paragraph({
      children: [
        new TextRun({ text: `Figure ${n}. `, font: FONT, size: 22, bold: true }),
        new TextRun({ text: caption, font: FONT, size: 22 }),
      ],
      spacing: { before: 60, after: 240, line: 320, lineRule: LineRuleType.AUTO },
      alignment: AlignmentType.LEFT,
    }),
  ];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const FONT = "Times New Roman";
const FONT_SIZE = 24; // 12pt in half-points
const DS = { line: 480, lineRule: LineRuleType.AUTO }; // double spacing
const A4_W = 11906; const A4_H = 16838;
const MARGIN = 1440; // 1 inch = 1440 dxa (use for all margins)
const CONTENT_W = A4_W - MARGIN * 2; // ~9026 dxa

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: FONT_SIZE, ...opts });
}
function bold(text) { return run(text, { bold: true }); }
function italic(text) { return run(text, { italics: true }); }

function para(children, opts = {}) {
  if (typeof children === "string") children = [run(children)];
  return new Paragraph({
    children,
    spacing: DS,
    ...opts,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: FONT, size: 26, bold: true })],
    spacing: { before: 240, after: 120 },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: FONT, size: 24, bold: true })],
    spacing: { before: 200, after: 80 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, italics: true })],
    spacing: { before: 160, after: 60 },
  });
}

function blankLine() {
  return new Paragraph({ children: [], spacing: DS });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()], spacing: { line: 240 } });
}

function figPlaceholder(n, filename, caption) {
  return [
    new Paragraph({
      children: [new TextRun({ text: `Figure ${n}. `, font: FONT, size: 22, bold: true }), new TextRun({ text: caption, font: FONT, size: 22 })],
      spacing: { before: 240, after: 200, line: 320, lineRule: LineRuleType.AUTO },
      alignment: AlignmentType.LEFT,
    }),
  ];
}

// ── Table helpers ─────────────────────────────────────────────────────────────

const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const NO_BORDER = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NO_BORDER, bottom: NO_BORDER, left: NO_BORDER, right: NO_BORDER };

function cell(runs, opts = {}) {
  if (typeof runs === "string") runs = [new TextRun({ text: runs, font: FONT, size: 20 })];
  return new TableCell({
    borders: BORDERS,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.TOP,
    width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
    shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      children: runs,
      spacing: { line: 320, lineRule: LineRuleType.AUTO },
      alignment: opts.align || AlignmentType.LEFT,
    })],
  });
}

function hCell(text, w) {
  return new TableCell({
    borders: BORDERS,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    width: w ? { size: w, type: WidthType.DXA } : undefined,
    shading: { fill: "2F5496", type: ShadingType.CLEAR },
    children: [new Paragraph({
      children: [new TextRun({ text, font: FONT, size: 20, bold: true, color: "FFFFFF" })],
      spacing: { line: 320, lineRule: LineRuleType.AUTO },
      alignment: AlignmentType.CENTER,
    })],
  });
}

// ── TABLE 1 ───────────────────────────────────────────────────────────────────────────────

function makeTable1() {
  const cols = [1900, 1400, 1400, 700, 1400, 1226];
  const totalW = cols.reduce((a, b) => a + b, 0);

  function row(model, intAUC, extAUC, gap, fairGap, brier, isHL) {
    isHL = isHL || false;
    const shade = isHL ? "E2EFDA" : undefined;
    const mkR = function(t, opts) { opts = opts || {}; return [new TextRun(Object.assign({ text: t, font: FONT, size: 20 }, opts))]; };
    return new TableRow({
      children: [
        cell(mkR(model, isHL ? { bold: true } : {}), { width: cols[0], shading: shade }),
        cell(mkR(intAUC), { width: cols[1], align: AlignmentType.CENTER, shading: shade }),
        cell(mkR(extAUC), { width: cols[2], align: AlignmentType.CENTER, shading: shade }),
        cell(mkR(gap), { width: cols[3], align: AlignmentType.CENTER, shading: shade }),
        cell(mkR(fairGap), { width: cols[4], align: AlignmentType.CENTER, shading: shade }),
        cell(mkR(brier), { width: cols[5], align: AlignmentType.CENTER, shading: shade }),
      ],
    });
  }

  const hdrs = ["Model", "Internal AUC [95% CI]", "External AUC [95% CI]", "Gen. Gap", "Fairness Gap [95% CI]", "Brier [95% CI]"];
  const headerRow = new TableRow({
    tableHeader: true,
    children: cols.map(function(w, i) { return hCell(hdrs[i], w); }),
  });

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      headerRow,
      row("XGBoost (Centralised)", "0.769 [0.760–0.777]", "0.700 [0.698–0.701]", "0.069", "0.069 [0.062–0.076]", "0.322 [0.321–0.322]"),
      row("DiabetesNet (Centralised, n=15,650)", "0.801 [0.782–0.819]", "0.749 [0.748–0.750]", "0.052", "0.063 [0.057–0.070]", "0.306 [0.305–0.306]"),
      row("FedAvg", "0.788 [0.780–0.796]", "0.757 [0.756–0.758]", "0.031", "0.054 [0.047–0.060]", "0.217 [0.217–0.218]", true),
      row("FedProx (μ=0.1)", "0.785 [0.777–0.793]", "0.752 [0.751–0.753]", "0.033", "0.066 [0.059–0.072]", "0.242 [0.241–0.242]"),
      row("FedNova (τ={5,3,4})", "0.786 [0.778–0.794]", "0.744 [0.743–0.745]", "0.042", "0.064 [0.058–0.071]", "0.266 [0.266–0.267]"),
      row("SCAFFOLD (SGD, 50 rds)†", "0.642 [0.631–0.653]†", "0.665 [0.664–0.667]", "−0.023‡", "−0.092 [−0.099–−0.085]", "0.187 [0.187–0.187]"),
      row("Sensitivity: DiabetesNet (matched subset, n=11,906)*", "0.795", "0.771†", "0.024", "0.063", "—"),
    ],
  });
}
// ── TABLE 2 ───────────────────────────────────────────────────────────────────

function makeTable2() {
  const cols = [2200, 1800, 1200, 3826];
  const totalW = cols.reduce((a, b) => a + b, 0);
  function row(method, ece_ci, auc, note, isHL = false) {
    const shade = isHL ? "E2EFDA" : undefined;
    const mkR = (t, opts = {}) => [new TextRun({ text: t, font: FONT, size: 20, ...opts })];
    return new TableRow({
      children: [
        cell(mkR(method, isHL ? { bold: true } : {}), { width: cols[0], shading: shade }),
        cell(mkR(ece_ci), { width: cols[1], align: AlignmentType.CENTER, shading: shade }),
        cell(mkR(auc), { width: cols[2], align: AlignmentType.CENTER, shading: shade }),
        cell(mkR(note), { width: cols[3], shading: shade }),
      ],
    });
  }
  const headerRow = new TableRow({
    tableHeader: true,
    children: cols.map((w, i) => hCell(["Method", "ECE [95% CI]", "AUC", "Notes"][i], w)),
  });
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      headerRow,
      row("FedAvg — Uncalibrated", "0.276 [0.275–0.277]", "0.757", "Moderate overconfidence; scores biased high"),
      row("FedAvg — Platt scaling", "0.007 [0.006–0.008]", "0.757", "Logistic regression on logit space; 2 parameters"),
      row("FedAvg — Isotonic regression", "0.001 [0.000–0.002]", "0.757", "Non-parametric; near-perfect calibration", true),
      row("FedAvg — Temperature (T = 2.41)", "0.269 [0.268–0.270]", "0.757", "Scalar logit transform; limited by non-monotone calibration curve"),
    ],
  });
}
// ── TABLE 3 (Related Work Comparison) ────────────────────────────────────────

function makeTable3() {
  const cols = [1200, 1100, 800, 900, 800, 700, 600, 600, 600, 726];
  const totalW = cols.reduce((a, b) => a + b, 0);
  function row(study, dataset, ntrain, next, nstrat, extval, fair, dp, cal, task) {
    return new TableRow({
      children: [study, dataset, ntrain, next, nstrat, extval, fair, dp, cal, task].map((t, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 40, bottom: 40, left: 60, right: 60 },
          width: { size: cols[i], type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: t, font: FONT, size: 18 })], spacing: { line: 280, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  const headers = ["Study", "Dataset", "n train", "n external", "# FL Strategies", "Ext. Valid.", "Fairness", "DP", "Calibration", "Task"];
  const headerRow = new TableRow({
    tableHeader: true,
    children: cols.map((w, i) => hCell(headers[i], w)),
  });
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      headerRow,
      row("Brisimi et al. 2018 [31]", "EHR (MIMIC)", "~5,600", "None", "1 (FedAvg)", "No", "No", "No", "No", "HF readmission"),
      row("Huang et al. 2019 [13]", "EHR (multi-site)", "~58,000", "None", "1 (federated)", "No", "No", "No", "No", "Mortality"),
      row("Sheller et al. 2020 [27]", "MRI (brain)", "~500", "Same-site", "1 (FedAvg)", "Limited", "No", "No", "No", "Tumour seg."),
      row("Dou et al. 2021 [12]", "CT (chest)", "~3,000", "Limited", "1 (FedAvg)", "Limited", "No", "No", "No", "COVID-19 det."),
      row("This work", "NHANES + BRFSS", "11,906", "1,282,897", "4 strategies", "Yes (1.28M)", "Yes", "Yes", "Yes", "Diabetes risk"),
    ],
  });
}

// ── BULLET PARA ───────────────────────────────────────────────────────────────
const bulletConfig = {
  reference: "bullets",
  levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
};
function bullet(children) {
  if (typeof children === "string") children = [run(children)];
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children, spacing: DS });
}

// ══════════════════════════════════════════════════════════════════════════════
// MANUSCRIPT DOCUMENT
// ══════════════════════════════════════════════════════════════════════════════

function buildManuscript(embedFigs = false) {
  const children = [];
  const P = (...args) => { children.push(para(...args)); };
  const B = (...args) => { children.push(bullet(...args)); };
  const H1 = t => children.push(h1(t));
  const H2 = t => children.push(h2(t));
  const H3 = t => children.push(h3(t));
  const BLANK = () => children.push(blankLine());
  const PAGE = () => children.push(pageBreak());
  const FIG = embedFigs
    ? (n, file, cap) => figImage(n, file, cap).forEach(p => children.push(p))
    : (n, file, cap) => figPlaceholder(n, file, cap).forEach(p => children.push(p));
  const TABLE = tbl => children.push(tbl);
  const NOTE = text => children.push(new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 20, italics: true })],
    spacing: { before: 80, after: 80, line: 320, lineRule: LineRuleType.AUTO },
  }));

  // ── TITLE PAGE ──────────────────────────────────────────────────────────────
  children.push(new Paragraph({
    children: [new TextRun({ text: "Privacy-Preserving Federated Learning for Diabetes Risk Prediction Across Demographically Heterogeneous Hospital Nodes", font: FONT, size: 28, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 240, line: 480, lineRule: LineRuleType.AUTO },
  }));
  BLANK();
  children.push(new Paragraph({
    children: [new TextRun({ text: "Rajveer Singh Pall", font: FONT, size: 24 })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Department of Computer Science and Engineering, Gyan Ganga Institute of Technology and Sciences, Jabalpur, Madhya Pradesh, India", font: FONT, size: 24 })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Email: rajveerpall04@gmail.com", font: FONT, size: 24 })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  BLANK();
  children.push(new Paragraph({
    children: [new TextRun({ text: "Submitted to: Computer Methods and Programs in Biomedicine (CMPB), Elsevier", font: FONT, size: 24, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  PAGE();

  // ── HIGHLIGHTS ──────────────────────────────────────────────────────────────
  H1("Highlights");
  B("FedAvg achieves AUC 0.757 [0.756–0.758] on 1.28 million BRFSS respondents.");
  B("FedAvg reduces elderly–young fairness gap 21.7% vs. centralised baseline.");
  B("Federation reduces generalisation gap 40% vs. matched centralised architecture (0.031 vs. 0.052).");
  B("Isotonic recalibration eliminates FedAvg overconfidence (ECE: 0.276 → 0.001).");
  B("DP-SGD utility floor at ε ≥ 10 (AUC = 0.769); formal DP infeasible at per-node n ≈ 3,000–4,500.");
  BLANK();

  // ── STATEMENT OF SIGNIFICANCE ────────────────────────────────────────────────
  H1("Statement of Significance");
  const sigCols = [2000, 7026];
  const sigTotal = sigCols.reduce((a, b) => a + b, 0);
  const sigBorder = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const sigBorders = { top: sigBorder, bottom: sigBorder, left: sigBorder, right: sigBorder };
  function sigCell(text, isHeader = false, shade = undefined) {
    return new TableCell({
      borders: sigBorders,
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      width: { size: isHeader ? sigCols[0] : sigCols[1], type: WidthType.DXA },
      shading: shade ? { fill: shade, type: ShadingType.CLEAR } : undefined,
      children: [new Paragraph({
        children: [new TextRun({ text, font: FONT, size: 20, bold: isHeader })],
        spacing: { line: 320, lineRule: LineRuleType.AUTO },
      })],
    });
  }
  children.push(new Table({
    width: { size: sigTotal, type: WidthType.DXA },
    columnWidths: sigCols,
    rows: [
      new TableRow({ children: [sigCell("Category", true, "DCE6F1"), sigCell("Statement", true, "DCE6F1")] }),
      new TableRow({ children: [sigCell("What is already known"), sigCell("Machine learning models for diabetes prediction from national surveillance data (NHANES, BRFSS) reach AUC 0.70–0.75 but are trained on pooled data, conflicting with patient privacy regulations. Multi-strategy federated learning benchmarks on external validation cohorts exceeding 1 million records are lacking.")] }),
      new TableRow({ children: [sigCell("What this study adds"), sigCell("This study benchmarks four FL aggregation strategies (FedAvg, FedProx, FedNova, SCAFFOLD) on three demographically stratified nodes (simulated via demographic stratification), validating on 1,282,897 BRFSS respondents. FedAvg reduces the generalisation gap 40% relative to a matched centralised architecture (0.031 vs. 0.052), reduces the elderly fairness gap 21.7%, and achieves near-perfect calibration after isotonic recalibration (ECE: 0.276→0.001).")] }),
    ],
  }));
  BLANK();
  PAGE();

  // ── ABSTRACT ─────────────────────────────────────────────────────────────────
  H1("Abstract");
  P([bold("Objective. "), run("To develop and evaluate a privacy-preserving federated learning (FL) framework for diabetes risk prediction across three demographically heterogeneous hospital nodes (simulated via demographic stratification of NHANES 2013–2020), without centralising patient data.")]);
  P([bold("Methods. "), run("NHANES 2013–2020 data (n = 15,650) were partitioned into three nodes representing young urban, elderly rural, and mixed-metropolitan populations. Four FL aggregation strategies—FedAvg, FedProx (μ = 0.1), FedNova, and SCAFFOLD—were trained over 50 communication rounds. A shared global scaler prevented data-leakage across nodes. External validation used BRFSS 2020–2022 (n = 1,282,897). Fairness was assessed as the AUC gap between young (18–39) and elderly (≥60) respondents. Post-hoc calibration (Platt, isotonic, temperature) and differential privacy (DP-SGD, ε∈{0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, ∞}) were evaluated. A centralised DiabetesNet was trained on the full NHANES pool to isolate architectural from federation effects.")]);
  P([bold("Results. "), run("FedAvg achieved the highest external AUC of 0.757 [95% CI: 0.756–0.758], exceeding the centralised XGBoost baseline (0.700 [0.698–0.701]) by 0.057. The centralised DiabetesNet (same architecture) reached external AUC = 0.749 [0.748–0.750], with a generalisation gap of 0.052 versus 0.031 for FedAvg—a 40% reduction relative to the full-pool centralised baseline. A sensitivity analysis on a sample-matched centralised model (n = 11,906) yielded a gap of 0.024, indicating that data composition rather than federation accounts for part of this advantage; federation additionally preserves privacy and improves elderly fairness. The elderly fairness gap was reduced from 0.069 (centralised XGBoost baseline) to 0.054 under FedAvg, a 21.7% within-study improvement. Isotonic recalibration of FedAvg reduced ECE from 0.276 to 0.001. At ε ≤ 5, the DP model collapsed to AUC ≈ 0.50. Recovery emerged at ε = 10.0 (AUC = 0.769); we characterise this as the DP-SGD utility floor for the per-node sample sizes used here (n ≈ 3,000–4,500), noting that Opacus RDP accounting returns ε_actual = ∞ at this budget and that formal cryptographic privacy guarantees would require substantially larger per-node cohorts.")]);
  P([bold("Conclusions. "), run("Federated learning improves generalisation, demographic equity, and calibration in diabetes risk prediction while preserving patient privacy. Federation reduces the generalisation gap relative to a full-pool centralised baseline (0.031 vs. 0.052), although a sample-matched sensitivity analysis indicates that data composition contributes substantially to this advantage; federation provides additional benefits—privacy preservation, improved elderly fairness, and feasibility under regulatory constraints—not captured by data composition alone. In the elderly subgroup (age ≥60, prevalence 20.2%), FedAvg achieves NNS = 8.6 at the Youden-optimal threshold versus XGBoost NNS = 8.0; at a sensitivity-prioritising threshold of 0.35 both converge to NNS = 5.1. The framework provides a validated proof-of-concept for privacy-preserving diabetes screening; prospective evaluation in genuinely independent multi-site settings remains an essential next step.")]);
  BLANK();
  P([bold("Keywords: "), run("federated learning; diabetes risk prediction; differential privacy; fairness-aware machine learning; calibration; NHANES; BRFSS")]);
  PAGE();

  // ── 1. INTRODUCTION ───────────────────────────────────────────────────────────
  H1("1. Introduction");
  P("Diabetes mellitus affects approximately 537 million adults worldwide and remains one of the leading causes of cardiovascular disease, renal failure, and premature mortality [1]. Early identification of at-risk individuals is therefore a global public health priority. Machine learning models trained on large electronic health record (EHR) cohorts have demonstrated promising discrimination in diabetes risk stratification [2], yet clinical adoption is impeded by two structural barriers: privacy regulation and demographic bias. In multi-hospital screening programs—the intended deployment setting for such models—centralised data pooling is rarely feasible, motivating privacy-preserving alternatives.");
  P("Centralised model training requires aggregating raw patient data from multiple institutions, which conflicts with HIPAA and GDPR. Even when data sharing is legally permissible, institutional reluctance to expose patient records limits the diversity and scale of training cohorts. Federated learning (FL) addresses this by distributing gradient computation to local nodes, sharing only model updates rather than raw data [3, 4]. Demographic heterogeneity across participating nodes introduces client drift—the divergence of local optima from the global objective—and can amplify disparities in model performance across age, sex, and socioeconomic strata [5]. Without explicit fairness constraints, a global model may optimise aggregate performance at the expense of the underrepresented subpopulations that most need accurate screening.");
  P("Prior diabetes prediction studies using national surveillance data have not simultaneously addressed privacy preservation, algorithmic fairness, and calibration within a federated paradigm, nor separated the contribution of federation from model architecture choice. Existing FL frameworks for clinical prediction (e.g., FedEnTrust [7], FeTS [8]) have not benchmarked multiple aggregation strategies head-to-head on the same demographically partitioned cohort against a matched centralised architecture, nor validated on a large independent external cohort.");
  P([run("This study makes four primary contributions. First, we implement and compare four FL aggregation strategies—FedAvg [3], FedProx [9], FedNova [10], and SCAFFOLD [11]—under identical conditions on three demographically stratified hospital nodes derived from NHANES, alongside a matched centralised DiabetesNet baseline to isolate the effect of federation from architecture. Second, we evaluate external validity at population scale on BRFSS 2020–2022 (n = 1,282,897)—to our knowledge, one of the largest independent external validations reported for this prediction task. Third, we quantify the privacy–utility tradeoff under differential privacy across an extended epsilon range (ε∈{0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, ∞}) to characterise the recovery boundary for clinical deployment. Fourth, we analyse post-hoc calibration and demographic fairness to assess clinical trustworthiness beyond discrimination alone.")]);

  // ── 2. BACKGROUND ─────────────────────────────────────────────────────────────
  H1("2. Background and Related Work");
  H2("2.1 Federated Learning in Healthcare");
  P("Federated learning, introduced by McMahan et al. [3] as FedAvg, enables collaborative model training without raw data exchange. The server broadcasts global model weights; each participating node performs local gradient descent and returns updated parameters; the server aggregates updates by weighted averaging proportional to local sample size. Applied to clinical prediction, FL has demonstrated utility in detecting COVID-19 from chest CT [12], predicting hospital readmission [13], and identifying rare phenotypes in genomic data [14].");
  P("Statistical heterogeneity remains the central challenge in federated healthcare settings: nodes corresponding to distinct demographic populations have divergent local distributions. FedProx [9] addresses this by augmenting the local objective with a proximal term (μ∥w−wᴵ∥²) that penalises deviation from the global model. FedNova [10] corrects for varying numbers of local updates through gradient normalisation. SCAFFOLD [11] introduces per-client control variates to remove client drift at the gradient level, with theoretical convergence guarantees under non-i.i.d. conditions.");
  P("Prior work on federated learning for diabetes prediction has primarily used smaller cohorts or has not evaluated multiple aggregation strategies under identical conditions [31]. The present study addresses both gaps by benchmarking four strategies on 1.28 million external respondents.");
  H2("2.2 Fairness-Aware Federated Learning");
  P("Algorithmic bias in clinical machine learning is well documented. Obermeyer et al. [15] demonstrated that commercially deployed risk scores systematically underestimate care needs of Black patients. In federated settings, client heterogeneity can either exacerbate or mitigate demographic disparities depending on data composition and aggregation strategy [16]. Fair federated learning approaches include q-FedAvg [30], which minimises the maximum loss across clients, and FairFML [18], which uses adversarial debiasing within the federation. Fairness is evaluated here post-hoc via subgroup AUC stratification, providing a measurement of inherent strategy-level differences without altering the training objective.");
  H2("2.3 Calibration in Risk Prediction");
  P("A risk score that correctly ranks patients (discrimination, measured by AUC) may still be miscalibrated—assigning a 70% risk estimate to a cohort where only 20% develop the outcome. For clinical decision support, calibration is arguably as important as discrimination [19]. The expected calibration error (ECE) measures the mean absolute gap between predicted probability and empirical event frequency across probability bins. Post-hoc methods including Platt scaling [20], isotonic regression [21], and temperature scaling [22] can substantially reduce ECE without retraining the underlying model.");

  H2("2.4 Positioning Relative to Prior Work");
  P("Table 3 summarises the four most directly comparable prior studies alongside this work across ten dimensions. No prior study simultaneously benchmarks four FL aggregation strategies, validates on an external cohort exceeding 1 million records, assesses demographic fairness, applies differential privacy, and reports post-hoc calibration. This work addresses all five dimensions.");
  BLANK();
  NOTE("Table 3. Related work comparison: federated learning for clinical prediction. Ext. Valid. = external validation on an independent cohort; DP = differential privacy evaluated.");
  TABLE(makeTable3());
  BLANK();

  // ── 3. METHODS ────────────────────────────────────────────────────────────────
  H1("3. Methods");
  H2("3.1 Data Sources and Feature Harmonisation");
  P("Training data were drawn from the National Health and Nutrition Examination Survey (NHANES) 2013–2020 (n = 15,650 after exclusion of records with >20% missing feature values). Records were excluded if any of the eight model features had more than 20% missingness; the remaining records had <5% missing values per feature. Missing values were handled by median imputation for continuous features (age, BMI) and mode imputation for categorical features (sex, race/ethnicity, smoking, physical activity, MI history, stroke history), applied within the NHANES training split to prevent leakage. Eight features were selected for cross-dataset compatibility: age, sex, race/ethnicity, body mass index (BMI), smoking status, physical activity level, history of myocardial infarction, and history of stroke. A StandardScaler was fitted exclusively on the NHANES training split (80/20 stratified, RANDOM_SEED = 42) and serialised for deployment-time transformation, applied transform-only at each federated node to prevent data leakage.");
  P("External validation used BRFSS 2020–2022 [6] (n = 1,282,897 after quality filtering). BRFSS features were harmonised to the eight NHANES features using a validated column mapping (Supplementary Table S1). Smoking was mapped from cigarette-days-per-year to a binary indicator; physical activity from self-reported minutes to a categorical level. Measurement invariance across surveys introduces residual noise in the smoking variable, as noted in limitations.");
  H2("3.2 Node Partitioning");
  P("NHANES records were partitioned into three nodes simulating distinct hospital catchment populations (Supplementary Table S3):");
  B([bold("Node A (Young Urban):"), run(" participants aged 18–44 or minority racial/ethnic groups (RIDRETH3 ∈ {Other Hispanic, Non-Hispanic Black}), n = 4,500 (training: 3,600)")]);
  B([bold("Node B (Elderly Rural):"), run(" participants aged ≥55 and White or Mexican-American, n = 3,406 (training: 2,725)")]);
  B([bold("Node C (Mixed Metropolitan):"), run(" remaining participants, n = 4,000 (training: 3,200)")]);
  P("This partition mirrors realistic demographic heterogeneity across rural and urban facilities. The three nodes together comprise n = 11,906 of the 15,650 NHANES records; the remaining 3,744 records are retained in the centralised dataset for baseline evaluation but are not assigned to any federated node. The 80/20 train/validation split was applied independently within each node using stratified sampling. Partitioning a single national survey by age simulates demographic heterogeneity but does not reproduce independent data collection processes or batch effects present in genuine multi-institutional deployments; results should be interpreted as proof-of-concept under controlled non-IID simulation.");
  P("To quantify the statistical heterogeneity induced by this partitioning, we computed the Jensen–Shannon divergence (JSD) between each pair of node label distributions. Node A (prevalence ~8%) versus Node B (prevalence ~28%) yields JSD = 0.035, confirming substantial distributional divergence. Node B versus Node C (prevalence ~15%) yields JSD = 0.013. These values motivated the heterogeneous τ assignments in FedNova (τ_A=5, τ_B=3, τ_C=4), where nodes with higher distributional shift receive fewer local update steps to reduce client drift.");
  P("To address label imbalance across nodes, class-weighted binary cross-entropy was applied during local training. Node A (prevalence ~8%) used positive class weight ~11.5; Node B (prevalence ~28%) used ~2.6; Node C (prevalence ~15%) used ~5.7. Class weighting prevents majority-class dominance in nodes with very low diabetes prevalence and is applied uniformly across all FL strategies.");
  H2("3.3 Neural Network Architecture");
  P("Both federated and centralised neural network experiments employed DiabetesNet: a four-layer fully connected network with architecture Input(8)→Dense(64, BatchNorm, ReLU, Dropout(0.3))→Dense(32, BatchNorm, ReLU, Dropout(0.3))→Dense(16, BatchNorm, ReLU, Dropout(0.18))→Dense(1, logit output). Binary cross-entropy with class-weighted positive class was minimised using AdamW (lr = 10⁻³, λ = 10⁻⁴) with CosineAnnealingLR scheduling. Total parameters: 4,993. All hyperparameters are documented in Supplementary Table S2.");
  FIG(1, "fig1_architecture.png", "Privacy-preserving federated learning system architecture. Three demographically distinct hospital nodes (Node A: Young Urban, Node B: Elderly Rural, Node C: Mixed Metropolitan) train local DiabetesNet models and exchange only parameter updates with a central aggregator. No raw patient records leave any node. The DiabetesNet architecture (right panel) shows the 4-layer MLP with batch normalisation and dropout regularisation at each hidden layer.");
  H2("3.4 Federated Aggregation Strategies");
  P("Four FL strategies were evaluated over 50 communication rounds (FL_NUM_ROUNDS = 50), each with 5 local epochs per round (NN_LOCAL_EPOCHS = 5) and batch size 256.");
  B([bold("FedAvg [3]:"), run(" weighted average of client parameters, proportional to local sample size. Serves as the canonical baseline.")]);
  B([bold("FedProx [9] (μ = 0.1):"), run(" augments the local objective with μ/2∥w−wᴵ∥², constraining client updates to stay near the global model. μ = 0.1 was selected by three-way grid search over {0.01, 0.1, 0.5} on internal validation AUC (documented in Supplementary Table S2).")]);
  B([bold("FedNova [10]:"), run(" normalises local gradients by the effective number of local update steps (τ∈{5, 3, 4} per node), correcting for heterogeneous local computation.")]);
  B([bold("SCAFFOLD [11] (Option II):"), run(" introduces per-client control variates to subtract client drift from local gradients. Implemented with SGD to match its theoretical derivation; optimizer difference relative to AdamW-based strategies is discussed in Section 5.2.")]);
  P("The comparison therefore reflects strategy-plus-optimizer behaviour under canonical implementations rather than a pure aggregation-only ablation; a matched AdamW-SCAFFOLD experiment would isolate the algorithmic contribution and is identified as a priority extension.");
  H2("3.5 Differential Privacy");
  P("Differential privacy was applied using DP-SGD [23] (Opacus framework) with Rényi Differential Privacy (RDP) accounting. An extended sweep of nine privacy budgets was evaluated: ε∈{0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, ∞} with fixed δ = 10⁻⁵. The clipping norm C = 1.0 and batch size 512 were held constant across all ε levels. The adversary model assumes a semi-honest server that can observe all model updates but does not collude with other clients, consistent with the standard FL threat model [24].");
  H2("3.6 Calibration Analysis");
  P("Post-hoc calibration was evaluated on FedAvg and FedProx predictions on the BRFSS external test set, using a 20/80 calibration/test split (n_cal = 256,579; n_test = 1,026,318). Three methods were compared: Platt scaling (logistic regression on logit space), isotonic regression, and temperature scaling (L-BFGS optimisation of NLL over a scalar T). ECE was computed using 10 equal-width probability bins with Wilson 95% confidence intervals per bin. Subgroup ECE was computed separately for young (18–39) and elderly (≥60) respondents on the NHANES internal validation set (n = 3,130).");
  H2("3.7 Statistical Analysis");
  P("Internal AUC 95% confidence intervals were computed by stratified bootstrap (N = 2000 resamples, percentile method). External AUC CIs used the DeLong structural components estimator [25] (O(n log n) implementation), computationally tractable for the 1.28 million BRFSS records. Pairwise DeLong tests were used for all FL strategy comparisons on the external set, with Bonferroni correction for three simultaneous comparisons (α = 0.017). For fairness metrics, 95% bootstrap confidence intervals (N = 2,000 resamples) were computed for all subgroup AUC gaps to quantify sampling uncertainty in the demographic disparity estimates. All analyses used RANDOM_SEED = 42. Implementation used Python 3.13, PyTorch 2.6.0, scikit-learn 1.6.1, and Opacus 1.4.0 for DP-SGD. This study follows the TRIPOD reporting guideline [33]; a TRIPOD compliance checklist is provided as Supplementary Appendix A. Brier score 95% CIs were computed by non-parametric bootstrap (N = 2,000, BRFSS external test set, RANDOM_SEED = 42).");

  // ── 4. RESULTS ────────────────────────────────────────────────────────────────
  H1("4. Results");
  H2("4.1 Training Convergence");
  P("All four FL strategies converged over 50 rounds on the NHANES internal validation set (Figure 2). FedAvg reached a final AUC of 0.788, with rapid early-round improvement (AUC = 0.764 at round 2) followed by gradual gains. FedProx and FedNova showed closely parallel trajectories (final AUC = 0.785 and 0.786, respectively), consistent with their shared AdamW optimiser and proximal regularisation. SCAFFOLD converged more slowly and reached a lower plateau (AUC = 0.642 at round 50), attributable to its SGD-based implementation (Section 5.2). A convergence criterion of ΔAUC < 0.001 over five consecutive rounds was reached by round 38 for FedAvg.");
  FIG(2, "fig2_convergence.png", "Convergence of four federated aggregation strategies over 50 communication rounds (NHANES internal validation AUC). FedAvg, FedProx, and FedNova converge to AUC ≊0.785–0.788 by round 38; SCAFFOLD (SGD) plateaus at 0.642. (FedAvg, FedProx, FedNova use AdamW; SCAFFOLD uses SGD — see Section 5.2 for optimizer interaction discussion.)");
  H2("4.2 Internal and External Validation");
  P("Table 1 summarises discrimination performance across all models. On the internal NHANES test set, all FL strategies outperformed both centralised baselines. FedAvg achieved the highest internal AUC of 0.788 [0.779–0.796]. On the external BRFSS validation set: FedAvg external AUC = 0.757 [0.756–0.758], FedProx 0.752 [0.751–0.753], FedNova 0.744 [0.743–0.745], and XGBoost centralised 0.700 [0.698–0.701].");
  P("To isolate the contribution of federation from neural network architecture, a centralised DiabetesNet trained on the full NHANES pool (n = 15,650) achieved an external AUC of 0.749 [0.748–0.750]—below all federated strategies despite access to 24% more training data (15,650 vs. 11,906). The critical distinction lies in the generalisation gap (internal − external AUC): FedAvg 0.031 vs. centralised DiabetesNet 0.052 vs. XGBoost 0.069. Federation reduces the generalisation gap by 40% relative to the matched architecture, indicating that the performance advantage reflects cross-dataset stability—not architecture—as the primary mechanism.");
  P("To assess the contribution of data composition to the observed generalisation advantage, a sensitivity analysis trained a centralised DiabetesNet on the combined federated nodes (matched subset: n = 11,906; n_train = 9,524; 80/20 stratified split) rather than the full NHANES pool. The matched-subset centralised model achieved internal AUC = 0.795 and external AUC = 0.771 (evaluated on n = 720,161 BRFSS respondents), yielding a generalisation gap of 0.024. This gap is modestly smaller than FedAvg’s gap of 0.031 (δ = 0.007), indicating that data composition accounts for part of the generalisation advantage. The result should be interpreted with caution: the BRFSS evaluation sample for this sensitivity analysis (n = 720,161) differs from the main analysis (n = 1,282,897), and the matched federated subset has a higher elderly prevalence than the full NHANES pool, which may independently reduce the external gap. Federation continues to provide additional benefits not captured by data composition alone: privacy preservation through local-only gradient exchange, improved demographic fairness (elderly gap 0.054 vs. 0.063 for the matched centralised model), and practical feasibility in settings where centralised data pooling is legally or operationally prohibited.");
  P("All pairwise FL–centralised comparisons were statistically significant (paired DeLong p < 0.001). Pairwise DeLong comparisons between FL strategies on BRFSS revealed that all three pairs were statistically distinguishable after Bonferroni correction: FedAvg vs. FedProx (ΔAUC = +0.005, z = 27.0, p < 0.001); FedAvg vs. FedNova (ΔAUC = +0.013, z = 72.5, p < 0.001); FedProx vs. FedNova (ΔAUC = +0.008, z = 36.5, p < 0.001). At n = 1.28 million, even a 0.005 AUC difference is statistically detectable; the absolute differences should be interpreted alongside clinical significance (Supplementary Table S4).");
  P("See Table 1 and Figures 3 and 7.");
  BLANK();
  NOTE("Table 1. Discrimination performance on internal (NHANES, n = 15,650) and external (BRFSS 2020–2022, n = 1,282,897) test sets. Two centralised baselines are reported to isolate the architecture and federation effects. Gen. Gap = Internal AUC − External AUC (lower is better). Shaded rows: FL strategies. SCAFFOLD Gen. Gap not computed due to optimizer mismatch (see † footnote). * Sensitivity analysis row: matched-subset centralised DiabetesNet (n=11,906); gap=0.024 < FedAvg gap=0.031.");
  TABLE(makeTable1());
  NOTE("† SCAFFOLD elderly gap is negative (elderly subgroup outperforms young), attributed to SGD convergence dynamics under heterogeneous nodes rather than a genuine fairness advantage; see Section 4.6.");
  NOTE("* Matched-subset sensitivity (Table 1, bottom row): centralised DiabetesNet trained on the combined federated nodes (n = 11,906; n_train = 9,524). External AUC evaluated on n = 720,161 BRFSS respondents. The matched-subset gap (0.024) is modestly smaller than FedAvg's gap (0.031, delta = 0.007), suggesting data composition partially explains the generalisation advantage; federation additionally provides privacy preservation and improved demographic fairness. No CIs reported for this row (sensitivity analysis only).");
  NOTE("CI: confidence interval; internal CI by stratified bootstrap (N = 2,000); external CI by DeLong structural components estimator. All pairwise FL strategy DeLong tests significant at Bonferroni-corrected α = 0.017.");
  BLANK();
  FIG(3, "fig3_roc_curves.png", "Receiver operating characteristic (ROC) curves for all models. Left: internal NHANES test set (n = 15,650). Right: external BRFSS validation set (n = 1,282,897). Shaded regions: 95% confidence intervals. FedAvg (blue) achieves the highest external AUC across both panels.");
  FIG(7, "fig7_generalisation_gap.png", "Generalisation gap between internal (NHANES) and external (BRFSS) validation AUC for all models. Error bars: 95% CI. Annotations: ΔAUC = internal−external. FedAvg generalisation gap (0.031) is 40% smaller than the centralised DiabetesNet (0.052). (Gap = Internal AUC − External AUC; lower gap indicates greater population robustness; FedAvg trained on n = 11,906 vs. centralised models on n = 15,650.)");

  H2("4.3 Demographic Fairness");
  P("Federated training consistently reduced the elderly–young AUC gap on BRFSS (Figure 4). The centralised XGBoost model yielded an elderly gap of 0.069 (young AUC = 0.656, elderly AUC = 0.587). FedAvg reduced this to 0.054 (young 0.722, elderly 0.669), a relative improvement of 21.7%. This within-study reduction maps directly to missed diagnoses: at a fixed screening threshold, a gap of 0.069 implies higher false-negative rates for elderly patients who already face elevated complication risk from undetected diabetes. FedProx and FedNova showed intermediate fairness gaps of 0.066 and 0.064. The sex-based gap was negligible across all strategies (ΔAUC < 0.01), and the BMI-stratified gap was reduced from 0.066 (centralised) to 0.016 (FedAvg).");
  P("DeLong 95% confidence intervals for the elderly–young AUC gap confirm statistical reliability: centralised XGBoost [0.062–0.076], FedAvg [0.047–0.060]. The non-overlapping intervals indicate that the 21.7% gap reduction is not attributable to sampling variance.");
  P("In the elderly subgroup (age ≥60, prevalence 20.2%, n = 578,859), the Number Needed to Screen (NNS = 1/[sensitivity × prevalence]) provides a clinically actionable summary of screening efficiency. At the Youden-optimal threshold, FedAvg (threshold = 0.656) achieves sensitivity = 0.577, specificity = 0.672, and NNS = 8.6 (approximately 9 elderly individuals screened per diabetes case detected). XGBoost (threshold = 0.757) achieves sensitivity = 0.618, specificity = 0.517, and NNS = 8.0. XGBoost detects more true positives per 10,000 screened at the Youden threshold (+82 TPs vs. FedAvg), though its lower specificity (0.517 vs. 0.672) implies 31% more false positives, which would trigger unnecessary follow-up investigations. At a clinically motivated lower threshold of 0.35 (prioritising sensitivity), both models converge to NNS = 5.1 (FedAvg sensitivity = 0.975, XGBoost sensitivity = 0.972), confirming comparable screening efficiency when recall is prioritised over specificity.");
  FIG(4, "fig4_fairness.png", "Fairness analysis on BRFSS external validation. (A) Age-stratified AUC for young (18–39) vs. elderly (≥60) respondents. (B) Fairness gap (ΔAUC = Young−Elderly) for each strategy. FedAvg reduces the elderly gap from 0.069 (centralised XGBoost) to 0.054 (21.7% within-study improvement). Bootstrap 95% CIs shown; non-overlapping intervals confirm statistical reliability.");

  H2("4.4 Calibration");
  P("The uncalibrated FedAvg model showed ECE = 0.276 on the BRFSS external test set (n_test = 1,026,318), consistent with the overconfidence pattern observed for FedProx (ECE = 0.319). Both federated strategies exhibit similar overconfidence, indicating that miscalibration reflects a framework-level distributional shift, not a strategy-specific artefact. Post-hoc calibration of FedAvg (Table 2) markedly improved reliability: Platt scaling reduced ECE to 0.007 while preserving discrimination (AUC unchanged at 0.757). Isotonic regression achieved ECE = 0.001, virtually eliminating miscalibration. Temperature scaling (T = 2.41) showed limited improvement (ECE = 0.269), because the calibration curve is non-monotonic. FedProx calibration results are provided in Supplementary Table S5 for comparison.");
  P("Subgroup analysis indicated that elderly respondents were more severely miscalibrated than young (ECE = 0.304 vs. 0.045). Subgroup ECE was computed on the NHANES internal validation set (n = 3,130) rather than the BRFSS calibration split because the 20% BRFSS calibration hold-out (n = 256,579), while large in aggregate, has insufficient elderly subgroup density within individual probability bins for stable ECE estimation at the 10-bin resolution used here. Platt scaling reduced both subgroup ECEs to <0.04.");
  P("See Table 2 and Figure 6.");
  BLANK();
  NOTE("Table 2. Calibration results for FedAvg (recommended strategy) on BRFSS external test set (n_test = 1,026,318; 20% calibration split). FedProx calibration results are provided in Supplementary Table S5. ECE: expected calibration error (10 equal-width bins). AUC is preserved across all calibration methods (post-hoc calibration does not alter rank ordering).");
  TABLE(makeTable2());
  BLANK();
  FIG(6, "fig6_calibration.png", "Calibration reliability diagrams for FedAvg on the BRFSS external test set (n = 1,026,318). Panels show uncalibrated model (ECE = 0.276), Platt scaling (ECE = 0.007), temperature scaling (ECE = 0.269), and isotonic regression (ECE = 0.001). Perfect calibration is shown as the dashed diagonal. Isotonic regression achieves near-perfect alignment between predicted probabilities and observed event rates.");

  H2("4.5 Differential Privacy");
  P("Differential privacy experiments demonstrated a sharp privacy–utility tradeoff (Figure 5). At tight privacy budgets (ε ≤ 5), the model collapsed to random performance (AUC ≈ 0.50), consistent with the high noise multiplier required given the small per-round batch fraction (sampling_rate = 512 / ∼10,000 ≊5%). Extending the sweep to ε∈{10, 20, 50, 100} revealed that the model recovers to AUC = 0.769 at ε = 10.0. We characterise this as the utility floor for DP-SGD in this dataset regime—the budget at which DP noise ceases to dominate the gradient signal—rather than a deployment-grade privacy threshold. Achieving formal DP with non-trivial utility at the per-node sample sizes used here (n ≈ 3,000–4,500) would require substantially larger per-node cohorts (~40,000). Without DP (ε = ∞), the model reached AUC = 0.766, confirming that collapse is caused by DP noise rather than numerical instability.");
  P("We note that Opacus RDP accounting returns ε_actual = ∞ at ε_target ≥ 10, meaning that at these budgets the DP noise is negligible relative to the gradient signal and no formal privacy guarantee is provided in the cryptographic sense. The ε = 10 result therefore identifies the utility recovery boundary—the point at which DP overhead ceases to degrade model performance—rather than a deployment-grade privacy threshold. Achieving formal DP at per-node sizes of ∼3,000–4,500 would require either larger cohorts (≥40,000 per node) or alternative mechanisms such as local DP with shuffling [28] or warm-up pre-training on public data [29].");
  FIG(5, "fig5_dp_tradeoff.png", "Privacy–utility tradeoff under differential privacy (extended sweep: ε∈{0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, ∞}). DP-SGD with δ = 10⁻⁵, clipping norm C = 1.0, batch size 512. Recovery emerges at ε = 10.0 (AUC = 0.769). (ε_actual = ∞ returned by Opacus RDP at ε_target ≥ 10, indicating negligible DP noise; ε = 10 marks utility recovery boundary, not a formal privacy guarantee.)");

  H2("4.6 SCAFFOLD Performance");
  P("SCAFFOLD (Option II) achieved an internal validation AUC of 0.642 at round 50, well below the AdamW-based strategies. On external BRFSS validation, SCAFFOLD achieved AUC = 0.665 [0.664–0.667], indicating that its limited internal performance generalises to held-out data. The Brier score (0.187) indicates reasonable probability calibration relative to its discrimination level. The convergence trajectory was monotonically increasing but slow. The performance gap is attributed to the SGD optimiser, which is required for the theoretical guarantees but lacks the adaptive moment estimation of AdamW. This finding is further analysed in Section 5.2.");
  P("Two features of the SCAFFOLD result merit attention beyond the SGD vs. AdamW optimiser difference. First, the inverted elderly–young AUC gap (−0.092 [−0.099–−0.085], Table 1 and Supplementary Table S6) is not a fairness advantage: SCAFFOLD's low overall discrimination (young AUC = 0.521, elderly AUC = 0.613) means its elderly outperformance reflects near-random behaviour on the young subgroup rather than improved elderly prediction. Second, despite low AUC, SCAFFOLD's external Brier score (0.187 [0.187–0.187]) is the lowest among all six models. This is internally consistent: under SGD without adaptive moments, SCAFFOLD produces flatter probability outputs concentrated near the prevalence, which reduces Brier score (a metric that rewards regression to the base rate) while sacrificing rank-ordering capability. Together these observations indicate that the limitation is convergence under heterogeneous nodes with vanilla SGD, not a property of the SCAFFOLD aggregation strategy per se.");
  FIG(8, "fig8_summary_comparison.png", "Overall performance summary across strategies. Blue bars: internal AUC; green bars: external AUC; red bars: fairness gap (×3 for visibility). Annotations show exact AUC values. Generalisation gap (horizontal difference between blue and green bars) is smallest for FedAvg (0.031), confirming population robustness as the key distinguishing metric.");

  // ── 5. DISCUSSION ─────────────────────────────────────────────────────────────
  H1("5. Discussion");
  H2("5.1 Federated Learning Improves Generalisation and Fairness");
  P("A key indicator of cross-dataset robustness is the generalisation gap between internal and external AUC. FedAvg's gap is 0.031, smaller than the full-pool centralised DiabetesNet's 0.052 and the centralised XGBoost's 0.069. The sample-matched sensitivity analysis (Section 4.2) shows that a centralised DiabetesNet trained on the same federated cohort (n = 11,906) achieves a gap of 0.024, smaller than FedAvg by 0.007. This narrows the interpretation: federation does not, on its own, reduce the gap relative to a sample-matched centralised model; rather, the federated cohort's demographic composition, together with federation, jointly determines the cross-dataset stability observed here. Federation's additional contribution lies elsewhere: privacy preservation, demographic fairness, and operational feasibility in regulated multi-site settings. The reduction in gap relative to the full-pool centralised baseline is consistent with two mechanisms: demographically partitioned nodes enforce generalisation across young urban, elderly rural, and mixed metropolitan subpopulations simultaneously; and local weight regularisation (FedProx) constrains client drift toward the numerically dominant node. These mechanisms align with broader evidence that federated training improves cross-site generalisation in clinical imaging [26, 27].");
  P("The fairness improvement is similarly principled. Because Node B (elderly rural) participates as an equal training partner with 18% of total samples, the FL model receives direct gradient feedback from the underperforming subgroup. FedAvg reduces the elderly gap from 0.069 (centralised XGBoost) to 0.054—a 21.7% within-study improvement that is clinically meaningful. The NNS analysis further contextualises this: at the Youden threshold, FedAvg achieves NNS = 8.6 in the elderly subgroup versus XGBoost NNS = 8.0. XGBoost's lower NNS reflects higher sensitivity (0.618 vs. 0.577), but its specificity is lower (0.517 vs. 0.672), meaning FedAvg generates 31% fewer false positives per 10,000 screened—a relevant consideration in healthcare systems where follow-up investigations for false positives impose costs and patient burden. At a sensitivity-prioritising threshold of 0.35, both models achieve NNS = 5.1, confirming that the choice of threshold mediates the practical tradeoff between sensitivity and specificity more than the choice of model.");
  P("The NNS values reported in Section 4.3 provide a more direct clinical anchor than AUC alone: at the Youden-optimal threshold, FedAvg yields NNS = 8.6 versus XGBoost NNS = 8.0 in the elderly subgroup, while at a sensitivity-prioritising threshold of 0.35 both converge to NNS = 5.1. The clinical operating point therefore mediates the practical implications of the AUC difference more than the choice of model.");

  H2("5.2 SCAFFOLD and the Optimizer Interaction");
  P("SCAFFOLD was implemented with SGD to satisfy its theoretical convergence guarantees (Karimireddy et al. [11]), while FedAvg, FedProx, and FedNova used AdamW. The lower performance of SCAFFOLD (internal AUC = 0.642, external AUC = 0.665) therefore reflects a combined effect of the algorithm's gradient correction mechanism and the optimiser difference—two factors that cannot be disentangled without an AdamW-based SCAFFOLD ablation. A direct comparison with AdamW-based SCAFFOLD would isolate the algorithm's contribution and represents a valuable direction for future work. The SGD-based result is reported because it corresponds to the theoretically grounded implementation as originally specified.");
  P("SCAFFOLD's underperformance relative to FedAvg is noteworthy because it was specifically designed for heterogeneous (non-i.i.d.) data. Adaptive moment estimation (AdamW) provides consistently faster loss reduction per round on the tabular health data used here. This interaction between aggregation strategy and local optimiser is underexplored in the FL literature and represents a productive direction for future investigation. An optimiser-matched SCAFFOLD comparison (AdamW instead of SGD) would isolate the algorithmic gradient-correction contribution from the optimiser effect and is identified as a priority extension");

  H2("5.3 Calibration and Clinical Deployment");
  P("The severe initial miscalibration (FedAvg ECE = 0.276) underscores that AUC alone is insufficient for clinical risk tools. A clinician relying on the uncalibrated FedAvg output would systematically over-estimate risk, potentially triggering unnecessary interventions. Isotonic recalibration reduces ECE to 0.001—a clinically negligible gap between predicted and observed event rates—but requires a representative hold-out set from the deployment population. Platt scaling reduces ECE to 0.007 with only two parameters and is recommended for deployments where a large calibration set is unavailable.");

  H2("5.4 Differential Privacy: Recovery Boundary and Clinical Implications");
  P("The differential privacy results identify a utility floor rather than a deployment-grade privacy threshold. The extended sweep (ε∈{10, 20, 50, 100}) shows recovery to AUC = 0.769 at ε = 10.0; however, Opacus RDP accounting returns ε_actual = ∞ at these budgets for the per-node sample sizes used here, indicating negligible DP noise relative to the gradient signal and no formal privacy guarantee in the cryptographic sense. At tighter budgets (ε ≤ 5), complete model collapse occurs. This reflects the mathematical constraint of DP-SGD at small per-round sample fractions: a per-node training set of approximately 3,300 samples with batch size 512 implies sampling_rate ≊15%, requiring noise multiplier σ ≳ 5 to achieve ε = 1. Achieving tight DP guarantees (ε < 10) would require either larger per-node datasets (≥40,000 samples) or weaker DP notions such as local DP with shuffling [28] or approximate DP with warm-up pre-training on public data [29]. These findings are presented transparently because they are directly relevant to practitioners considering FL with DP in similarly sized healthcare datasets. It should be noted that Opacus RDP accounting returns ε_actual = ∞ at ε_target ≥ 10, meaning that at these budget levels the DP noise is negligible relative to the gradient signal and no formal privacy guarantee exists in the cryptographic sense; the ε = 10 result therefore identifies the utility recovery boundary rather than a deployment-grade privacy threshold.");

  H2("5.5 Limitations");
  B("Node partitioning was simulated from a single NHANES cohort; a real deployment would have genuinely independent data collection and batch effects not captured here.");
  B("The BRFSS smoking variable mapping introduces measurement error relative to the NHANES clinical measurement (Supplementary Table S1).");
  B("The diabetes outcome variable differs across datasets: NHANES uses a composite definition (self-report + HbA1c ≥ 6.5% + fasting glucose ≥ 126 mg/dL) while BRFSS relies on self-reported diagnosis only. BRFSS therefore captures diagnosed cases and misses undiagnosed diabetes, which is expected to attenuate the external AUC estimate (i.e., the reported external AUC is plausibly a lower bound on the true discrimination of the model against a clinically verified outcome). This direction of bias does not affect the within-BRFSS comparisons between strategies.");
  B("SCAFFOLD was implemented with SGD to satisfy its theoretical convergence requirements, while the other strategies used AdamW; the strategy-versus-optimiser contributions to its lower discrimination cannot be disentangled without a matched-optimiser ablation, which is identified as a priority extension.");
  B("Differential privacy was evaluated in a standalone DP-FL experiment; integrating per-round DP noise into FedAvg/FedProx training is a natural extension.");
  B("All nodes used the same DiabetesNet architecture; personalised FL methods (e.g., Ditto [17]) with heterogeneous models were not evaluated.");
  B("The sensitivity analysis (Section 4.2) reveals that the matched-subset centralised model achieves a slightly smaller generalisation gap (0.024) than FedAvg (0.031), suggesting data composition partially explains the advantage; a fully matched experimental design with genuinely independent data collection would provide stronger causal evidence.");
  B("Bootstrap confidence intervals for fairness gaps are reported (Section 4.3), but uncertainty quantification for calibration metrics (ECE) across demographic subgroups under deployment distribution shift is not addressed and warrants future study.");
  B("Brier score CIs were computed by non-parametric bootstrap on the BRFSS external set (N = 2,000); CI widths are narrow owing to large n, but absolute Brier values reflect dataset-level calibration and should be interpreted with the known BRFSS self-report bias in the diabetes outcome variable in mind.");
  B("Model explainability was not addressed in this study; a SHAP-based feature importance analysis [32] applied to the FedAvg model is a priority extension for clinical interpretability and for characterising whether each demographic node contributes coherently to the global decision surface.");

  // ── 6. CONCLUSIONS ────────────────────────────────────────────────────────────
  H1("6. Conclusions");
  P("Federated learning is a viable framework for diabetes risk prediction across demographically heterogeneous nodes, achieving discrimination on external data that matches or exceeds full-pool centralised baselines while preserving patient privacy. The generalisation gap of FedAvg (0.031) is smaller than that of the full-pool centralised DiabetesNet (0.052); a sample-matched sensitivity analysis shows that data composition explains part of this advantage (matched-subset gap = 0.024), while federation additionally contributes privacy preservation, demographic fairness, and operational feasibility under regulatory constraints. FedAvg achieves the best external AUC of 0.757 [0.756–0.758] on 1.28 million BRFSS respondents and reduces the elderly fairness gap from 0.069 (centralised baseline) to 0.054—a 21.7% within-study improvement. Post-hoc isotonic calibration eliminates model overconfidence (ECE: 0.276→0.001). Together, these results provide evidence for federated learning as a viable framework for privacy-preserving diabetes screening at scale.");
  P("Practical adoption requires attention to three open challenges: differential privacy is not formally achievable at the per-node sample sizes evaluated here (DP-SGD utility recovers at ε ≥ 10 but RDP accounting returns ε_actual = ∞ at these budgets); calibration must be re-fit on the deployment population because train–test distribution shift causes ECE to drift; and prospective evaluation in cohorts with systematically ascertained elderly outcomes is required to validate the fairness improvements. All code, hyperparameters, and supplementary materials are released to support reproducibility.");
  P("The framework constitutes a validated proof-of-concept under controlled demographic heterogeneity; translation to genuinely independent multi-site settings requires prospective evaluation with independent data collection and real institutional batch effects. In the near term, the validated FedAvg pipeline could support state-level BRFSS surveillance programmes where participating health departments retain local data while contributing to a shared screening model. Three extensions are identified as priorities: (i) an optimiser-matched SCAFFOLD ablation (AdamW instead of SGD) to isolate the algorithm's gradient-correction contribution from the optimiser effect; (ii) SHAP-based feature importance analysis [32] of the FedAvg model for clinical interpretability; and (iii) integration of per-round DP noise into FedAvg/FedProx with cohort sizes sufficient to provide non-trivial cryptographic guarantees.");

  // ── ACKNOWLEDGEMENTS ──────────────────────────────────────────────────────────
  H1("Acknowledgements");
  P("The author acknowledges the CDC (BRFSS) and NCHS (NHANES) for publicly releasing the datasets used in this study. No external funding was received. The author declares no competing financial or personal interests that could have influenced this work.");

  // ── DATA AVAILABILITY ─────────────────────────────────────────────────────────
  H1("Data Availability Statement");
  P([run("The NHANES data used in this study are publicly available from the National Center for Health Statistics at "), new ExternalHyperlink({ children: [new TextRun({ text: "https://www.cdc.gov/nchs/nhanes/", font: FONT, size: FONT_SIZE, style: "Hyperlink" })], link: "https://www.cdc.gov/nchs/nhanes/" }), run(". The BRFSS data are publicly available from the Centers for Disease Control and Prevention at "), new ExternalHyperlink({ children: [new TextRun({ text: "https://www.cdc.gov/brfss/", font: FONT, size: FONT_SIZE, style: "Hyperlink" })], link: "https://www.cdc.gov/brfss/" }), run(". All analysis code, trained model configurations, and hyperparameters are available at "), new ExternalHyperlink({ children: [new TextRun({ text: "https://github.com/Rajveer-code/Federated-Diabetes", font: FONT, size: FONT_SIZE, style: "Hyperlink" })], link: "https://github.com/Rajveer-code/Federated-Diabetes" }), run(". No IRB approval was required; all data are publicly available, de-identified national surveillance datasets.")]);

  // ── AUTHOR CONTRIBUTION ───────────────────────────────────────────────────────
  H1("Author Contribution Statement");
  P("Rajveer Singh Pall: Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization.");

  // ── REFERENCES ────────────────────────────────────────────────────────────────
  H1("References");
  const refs = [
    "[1] IDF Diabetes Atlas, 10th ed. International Diabetes Federation, 2021.",
    "[2] Tigga NP, Garg S. Prediction of type 2 diabetes using machine learning classification methods. Procedia Computer Science. 2020;167:706–716.",
    "[3] McMahan HB, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.",
    "[4] Rieke N, Hancox J, Li W, et al. The future of digital health with federated learning. npj Digital Medicine. 2020;3:119.",
    "[5] Li T, Diao S, Chen PH, et al. Federated learning on non-iid data silos. arXiv:2102.02079, 2021.",
    "[6] CDC. Behavioral Risk Factor Surveillance System (BRFSS). https://www.cdc.gov/brfss, 2022.",
    "[7] Antunes RS, André da Costa C, Küderle A, et al. Federated learning for healthcare: Systematic review and architecture proposal. Digital Health. 2022;8.",
    "[8] Pati S, Baid U, Edwards B, et al. The federated tumor segmentation (FeTS) challenge. arXiv:2105.05874, 2021.",
    "[9] Li T, Sahu AK, Zaheer M, et al. Federated optimization in heterogeneous networks. MLSys 2020.",
    "[10] Wang J, Liu Q, Liang H, et al. Tackling the objective inconsistency problem in heterogeneous federated optimization. NeurIPS 2020.",
    "[11] Karimireddy SP, Kale S, Mohri M, et al. SCAFFOLD: Stochastic controlled averaging for federated learning. ICML 2020.",
    "[12] Dou Q, So TY, Jiang M, et al. Federated deep learning for detecting COVID-19 lung abnormalities in CT. NPJ Digital Medicine. 2021;4:60.",
    "[13] Huang L, Shea AL, Qian H, et al. Patient clustering improves efficiency of federated machine learning to predict mortality and hospital stay time using distributed electronic health records. J Biomed Inform. 2019;99:103291.",
    "[14] Warnat-Herresthal S, Schultze H, Shastry KL, et al. Swarm learning for decentralized and confidential clinical machine learning. Nature. 2021;594:265–270.",
    "[15] Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366:447–453.",
    "[16] Ezzeldin YH, Yan S, He C, et al. FairFed: Enabling group fairness in federated learning. AAAI 2023.",
    "[17] Li T, Hu S, Beirami A, Smith V. Ditto: Fair and robust federated learning through personalization. ICML 2021.",
    "[18] Du M, Yang F, Zou N, Hu X. Fairness in deep learning: A computational perspective. IEEE Intelligent Systems. 2021;36:25–34.",
    "[19] Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: The Achilles heel of predictive analytics. BMC Medicine. 2019;17:230.",
    "[20] Platt J. Probabilistic outputs for support vector machines. In: Advances in Large Margin Classifiers, 1999.",
    "[21] Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. KDD 2002.",
    "[22] Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017.",
    "[23] Abadi M, Chu A, Goodfellow I, et al. Deep learning with differential privacy. CCS 2016.",
    "[24] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning. CCS 2017.",
    "[25] DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves. Biometrics. 1988;44:837–845.",
    "[26] Rajpurkar P, Chen E, Banerjee O, Topol EJ. AI in health and medicine. Nature Medicine. 2022;28:31–38.",
    "[27] Sheller MJ, Edwards B, Reina GA, et al. Federated learning in medicine. Scientific Reports. 2020;10:12598.",
    "[28] Erlingsson Ú, Pihur V, Korolova A. RAPPOR: Randomized aggregatable privacy-preserving ordinal response. CCS 2014.",
    "[29] Tramèr F, Boneh D. Differentially private learning needs better features. ICLR 2021.",
    "[30] Li T, Diao S, Chen PH, Karimireddy SP, Smith V. Fair resource allocation in federated learning. ICLR 2020.",
    "[31] Brisimi TS, Chen R, Mela T, Olshevsky A, Paschalidis IC, Shi W. Federated learning of predictive models from federated Electronic Health Records. International Journal of Medical Informatics. 2018;112:59–67.",
    "[32] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS). 2017;30.",
    "[33] Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. BMJ. 2015;350:g7594.",
  ];
  refs.forEach(r => P(r));

  // (author-internal scaffolding sections removed from submitted manuscript)

  return new Document({
    numbering: { config: [bulletConfig] },
    styles: {
      default: { document: { run: { font: FONT, size: FONT_SIZE } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 2 } },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: A4_W, height: A4_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
        lineNumbers: { countBy: 1, start: 1, restart: "newPage" },
      },
      headers: {
        default: new Header({ children: [new Paragraph({ children: [new TextRun({ text: "Pall — FL for Diabetes Risk Prediction — CMPB Submission", font: FONT, size: 20, italics: true })], alignment: AlignmentType.RIGHT })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ children: [new TextRun({ text: "Page ", font: FONT, size: 20 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 }), new TextRun({ text: " of ", font: FONT, size: 20 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 20 })], alignment: AlignmentType.CENTER })] }),
      },
      children,
    }],
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// COVER LETTER
// ══════════════════════════════════════════════════════════════════════════════

function buildCoverLetter() {
  const children = [];
  const P = (...args) => { children.push(para(...args)); };
  const BLANK = () => children.push(blankLine());

  P([run("19 May 2026", { bold: false })]);
  BLANK();
  P("The Editorial Board");
  P("Computer Methods and Programs in Biomedicine");
  P("Elsevier");
  BLANK();
  P([bold("Re: Submission of Original Research Article")]);
  BLANK();
  P("Dear Editors,");
  BLANK();
  P("I submit for your consideration an original research article entitled “Privacy-Preserving Federated Learning for Diabetes Risk Prediction Across Demographically Heterogeneous Hospital Nodes.” This manuscript addresses a critical gap in clinical machine learning: the simultaneous optimisation of privacy preservation, algorithmic fairness, and probabilistic calibration within a federated training framework applied to diabetes risk prediction.");
  BLANK();
  P("Diabetes prediction models trained on national surveillance data typically require centralised data pooling, which conflicts with patient-privacy regulations (HIPAA, GDPR); many also exhibit reduced performance in elderly and minority subgroups and are poorly calibrated. Federated learning (FL) offers a principled alternative, but multi-strategy benchmarks on independent external cohorts with formal fairness and calibration analysis are scarce.");
  BLANK();
  P("The contributions of this study are fivefold. First, FedAvg achieves an external AUC of 0.757 [95% CI 0.756–0.758] on 1,282,897 BRFSS respondents—one of the largest independent external validation cohorts reported for federated diabetes prediction. Second, a sample-matched sensitivity analysis disentangles the contribution of data composition from federation per se: FedAvg's generalisation gap (0.031) is smaller than the full-pool centralised DiabetesNet's (0.052) but slightly larger than a matched-subset centralised model's (0.024), so the federation advantage lies in privacy, fairness, and operational feasibility rather than cross-dataset stability alone. Third, federated training reduces the elderly–young AUC fairness gap from 0.069 (centralised baseline) to 0.054 (FedAvg), a 21.7% within-study improvement; isotonic recalibration further reduces FedAvg ECE from 0.276 to 0.001 while preserving discrimination. Fourth, an extended differential privacy sweep (ε∈{0.5, 1, 2, 5, 10, 20, 50, 100, ∞}) identifies a utility floor at ε = 10 (AUC = 0.769) for per-node n ≈ 3,000–4,500; we transparently note that Opacus RDP returns ε_actual = ∞ at this budget, so the result characterises the dataset-size constraint on DP feasibility rather than a deployment-grade privacy guarantee. Fifth, the study follows the TRIPOD reporting guideline (Collins et al. BMJ 2015); a full TRIPOD compliance checklist is provided as Supplementary Appendix A. Feature-importance analysis (SHAP) is identified as a priority extension in the limitations section.");
  BLANK();
  P("This work is a good fit for Computer Methods and Programs in Biomedicine (CMPB), combining computational methodology (four FL strategies, DP-SGD with Opacus RDP accounting, DeLong-based inference on n > 1.28 million) with clinical relevance (diabetes screening equity, calibrated probability outputs, transparency about DP feasibility). All code and hyperparameter specifications are publicly released; both training (NHANES) and external validation (BRFSS) datasets are open-access. The external validation cohort (n = 1,282,897) is, to our knowledge, among the largest reported for federated diabetes prediction.");
  BLANK();
  P("I confirm that this manuscript is original, has not been previously published in whole or in part, and is not under simultaneous consideration at another journal. All data analysed are publicly available from the U.S. CDC (NHANES and BRFSS), and all analysis code is publicly available at https://github.com/Rajveer-code/Federated-Diabetes. The study uses de-identified public surveillance data and does not require institutional ethics approval.");
  BLANK();
  P("Data and code availability: NHANES 2013–2020 and BRFSS 2020–2022 are publicly available from the U.S. CDC. All analysis code is publicly available at https://github.com/Rajveer-code/Federated-Diabetes.");
  BLANK();
  P("I declare no competing financial or personal interests that could have influenced this work. No external funding was received for this study.");
  BLANK();
  P("I hope this work will be of interest to readers of Computer Methods and Programs in Biomedicine and look forward to the reviewers’ comments.");
  BLANK();
  P("Sincerely,");
  BLANK();
  BLANK();
  P("Rajveer Singh Pall");
  P("Department of Computer Science and Engineering");
  P("Gyan Ganga Institute of Technology and Sciences");
  P("Jabalpur, Madhya Pradesh, India");
  P("Email: rajveerpall04@gmail.com");

  return new Document({
    styles: { default: { document: { run: { font: FONT, size: FONT_SIZE } } } },
    sections: [{
      properties: { page: { size: { width: A4_W, height: A4_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } } },
      footers: { default: new Footer({ children: [new Paragraph({ children: [new TextRun({ text: "Page ", font: FONT, size: 20 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })], alignment: AlignmentType.CENTER })] }) },
      children,
    }],
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SUPPLEMENTARY DOCUMENT
// ══════════════════════════════════════════════════════════════════════════════

function buildSupplementary() {
  const children = [];
  const P = (...args) => { children.push(para(...args)); };
  const H1 = t => children.push(h1(t));
  const H2 = t => children.push(h2(t));
  const BLANK = () => children.push(blankLine());
  const PAGE = () => children.push(pageBreak());
  const TABLE = tbl => children.push(tbl);
  const NOTE = text => children.push(new Paragraph({ children: [new TextRun({ text, font: FONT, size: 20, italics: true })], spacing: { before: 60, after: 60, line: 320, lineRule: LineRuleType.AUTO } }));

  // Header
  children.push(new Paragraph({
    children: [new TextRun({ text: "Supplementary Material", font: FONT, size: 32, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 120, line: 480, lineRule: LineRuleType.AUTO },
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Privacy-Preserving Federated Learning for Diabetes Risk Prediction Across Demographically Heterogeneous Hospital Nodes", font: FONT, size: 24, italics: true })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120, line: 360, lineRule: LineRuleType.AUTO },
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Computer Methods and Programs in Biomedicine (CMPB), Elsevier", font: FONT, size: 22 })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Supplementary Tables S1–S6", font: FONT, size: 22, bold: true })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Rajveer Singh Pall", font: FONT, size: 22 })],
    alignment: AlignmentType.CENTER,
    spacing: DS,
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "Gyan Ganga Institute of Technology and Sciences, Jabalpur, India", font: FONT, size: 22 })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 480, line: 360, lineRule: LineRuleType.AUTO },
  }));

  // Contents
  H1("Contents");
  ["Table S1 — Feature Harmonisation Mapping (NHANES ↔ BRFSS)",
   "Table S2 — Full Hyperparameter Specification",
   "Table S3 — Node Partition Details",
   "Table S4 — Pairwise DeLong Tests Between FL Strategies",
   "Table S5 — FedProx Post-Hoc Calibration on BRFSS",
   "Table S6 — DeLong Confidence Intervals for Demographic Fairness Gaps"].forEach(t => P(t));
  PAGE();

  // ── TABLE S1 ─────────────────────────────────────────────────────────────────
  H1("Table S1. Feature Harmonisation Mapping Between NHANES and BRFSS");
  P("Eight predictor features enter the DiabetesNet model; rows 1–8 below are the model inputs harmonised across NHANES 2013–2020 and BRFSS 2020–2022. Row 9 describes the diabetes outcome (prediction target).");
  BLANK();

  const s1Cols = [1500, 1600, 1600, 2226, 2100];
  const s1Total = s1Cols.reduce((a, b) => a + b, 0);
  function s1Row(feat, nhanes, brfss, notes, diff) {
    return new TableRow({
      children: [s1Cols[0], s1Cols[1], s1Cols[2], s1Cols[3], s1Cols[4]].map((w, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 60, bottom: 60, left: 80, right: 80 },
          width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: [feat, nhanes, brfss, notes, diff][i], font: FONT, size: 18 })], spacing: { line: 300, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  TABLE(new Table({
    width: { size: s1Total, type: WidthType.DXA },
    columnWidths: s1Cols,
    rows: [
      new TableRow({ tableHeader: true, children: s1Cols.map((w, i) => hCell(["Feature", "NHANES Variable", "BRFSS Variable", "Mapping Notes", "Known Measurement Differences"][i], w)) }),
      s1Row("Age", "RIDAGEYR (continuous, years)", "AGEG5YR (13 age groups, midpoint)", "BRFSS group midpoint used as continuous proxy", "Low: slight grouping error at group boundaries"),
      s1Row("Sex", "RIAGENDR (1=Male, 2=Female)", "_SEX (1=Male, 2=Female)", "Direct match; same coding", "Low: consistent definition across surveys"),
      s1Row("Race/Ethnicity", "RIDRETH3 (6 categories)", "_RACE (5 categories)", "NHANES 6-group to BRFSS 5-group alignment; Hispanic merged", "Medium: Hispanic sub-group definitions differ slightly"),
      s1Row("BMI", "BMXBMI (measured, kg/m²)", "_BMI5 / 100 (self-reported)", "BRFSS _BMI5 divided by 100 to restore kg/m² scale", "Medium: self-report underestimates ~1.5 kg/m² systematically"),
      s1Row("Smoking status", "SMQ020 / SMQ040 (ever / current smoker)", "SMOKE100 / SMOKDAY2 (100 cigarettes / current)", "Mapped to 3-level: never=0, former=1, current=2", "High: NHANES lab-verified cotinine; BRFSS self-reported; residual mismatch"),
      s1Row("Physical activity", "PAQ605 / PAQ620 / PAQ635 / PAQ650 / PAQ665 (any domain)", "EXERANY2 (any leisure activity, binary)", "Any domain reporting ‘Yes’ → active (1); else inactive (0)", "Medium: NHANES multi-domain minutes; BRFSS single binary flag"),
      s1Row("MI history", "MCQ160E (ever told MI, binary)", "CVDINFR4 (ever told MI, binary)", "Direct match; both self-reported history of MI", "Low: both self-reported; consistent coding"),
      s1Row("Stroke history", "MCQ160F (ever told stroke, binary)", "CVDSTRK3 (ever told stroke, binary)", "Direct match; both self-reported stroke history", "Low: both self-reported; consistent coding"),
      s1Row("Diabetes (outcome)", "DIQ010 + LBXGH + LBXGLU (composite)", "DIABETE4 (self-reported diagnosis)", "NHANES: composite of self-report, HbA1c ≥56.5%, fasting glucose ≥126 mg/dL", "Medium: NHANES clinically confirmed; BRFSS self-reported diagnosis only"),
    ],
  }));
  BLANK();
  NOTE("Sensitivity analysis: model retrained with smoking variable excluded showed ΔAUC < 0.003 on both internal and external validation sets, confirming that measurement mismatch in the smoking variable does not materially affect results.");
  PAGE();

  // ── TABLE S2 ─────────────────────────────────────────────────────────────────
  H1("Table S2. Full Hyperparameter Specification");
  P("All hyperparameters are reported as used in the final experiments. Hyperparameters marked with * were determined by grid search; others follow established defaults or theoretical recommendations.");
  BLANK();
  const s2Cols = [1800, 2200, 1800, 3226];
  const s2Total = s2Cols.reduce((a, b) => a + b, 0);
  function s2Row(comp, param, val, app) {
    return new TableRow({
      children: [s2Cols[0], s2Cols[1], s2Cols[2], s2Cols[3]].map((w, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 60, bottom: 60, left: 80, right: 80 },
          width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: [comp, param, val, app][i], font: FONT, size: 18 })], spacing: { line: 300, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  const s2Data = [
    ["DiabetesNet", "Input dimension", "8", "All FL and centralised models"],
    ["DiabetesNet", "Hidden layer sizes", "64 → 32 → 16", "All FL and centralised models"],
    ["DiabetesNet", "Dropout rate (layers 1–2)", "0.3", "All FL and centralised models"],
    ["DiabetesNet", "Dropout rate (layer 3)", "0.18", "All FL and centralised models"],
    ["DiabetesNet", "Batch normalisation", "Yes (after each hidden layer)", "All FL and centralised models"],
    ["DiabetesNet", "Output", "Dense(1) logit — BCEWithLogitsLoss", "All FL and centralised models"],
    ["DiabetesNet", "Total parameters", "4,993", "All FL and centralised models"],
    ["Optimiser", "Algorithm", "AdamW", "FedAvg, FedProx, FedNova, centralised NN"],
    ["Optimiser", "Learning rate (lr)*", "0.001 (grid-searched: {1e-4, 1e-3, 5e-3})", "FedAvg, FedProx, FedNova, centralised NN"],
    ["Optimiser", "Weight decay (λ)", "0.0001", "FedAvg, FedProx, FedNova, centralised NN"],
    ["Optimiser", "Scheduler", "CosineAnnealingLR (T_max = FL_NUM_ROUNDS)", "FedAvg, FedProx, FedNova, centralised NN"],
    ["Optimiser", "SCAFFOLD algorithm", "SGD (lr = 1e-3)", "SCAFFOLD only (required for theoretical guarantees)"],
    ["Training", "Batch size", "256", "All local node training"],
    ["Training", "Local epochs per round", "5", "FedAvg, FedProx, SCAFFOLD"],
    ["Training", "Mixed precision (AMP)", "Yes (CUDA) / No (CPU)", "All models"],
    ["Training", "Random seed", "42", "All experiments"],
    ["Federated", "Communication rounds", "50", "All FL strategies"],
    ["Federated", "Number of clients (nodes)", "3", "All FL strategies"],
    ["Federated", "FedProx proximal μ*", "0.1 (grid-searched: {0.01, 0.1, 0.5})", "FedProx only"],
    ["FedNova", "Node A local epochs (τ_A)", "5 (Young Urban: low shift)", "FedNova only"],
    ["FedNova", "Node B local epochs (τ_B)", "3 (Elderly Rural: high shift)", "FedNova only"],
    ["FedNova", "Node C local epochs (τ_C)", "4 (Mixed Metro: intermediate)", "FedNova only"],
    ["SCAFFOLD", "Option", "Option II (Karimireddy et al. ICML 2020)", "SCAFFOLD only"],
    ["SCAFFOLD", "Control variate steps K", "5 (= NN_LOCAL_EPOCHS)", "SCAFFOLD only"],
    ["SCAFFOLD", "Local learning rate η_l", "1e-3", "SCAFFOLD only"],
    ["SCAFFOLD", "Global learning rate η_g", "1.0 (Option II standard)", "SCAFFOLD only"],
    ["Diff. Privacy", "Framework", "Opacus ≥ 1.4 (DP-SGD)", "DP experiments only"],
    ["Diff. Privacy", "Accountant", "RDP (Rényi DP)", "DP experiments only"],
    ["Diff. Privacy", "Gradient clipping norm C", "1.0", "DP experiments only"],
    ["Diff. Privacy", "DP batch size", "512", "DP experiments only"],
    ["Diff. Privacy", "δ (failure probability)", "1e-05", "DP experiments only"],
    ["Diff. Privacy", "ε levels tested (extended sweep)", "0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, ∞", "DP experiments only"],
    ["Statistics", "Bootstrap resamples (internal CI)", "2,000", "AUC confidence intervals"],
    ["Statistics", "Bootstrap method", "Stratified percentile", "AUC confidence intervals"],
    ["Statistics", "External CI method", "DeLong structural components estimator", "BRFSS external AUC"],
    ["Statistics", "Calibration bins", "10 equal-width probability bins", "ECE computation"],
    ["Statistics", "Calibration split (BRFSS)", "20% calibration / 80% test", "Post-hoc calibration (FedAvg and FedProx)"],
    ["Statistics", "Pairwise comparisons", "3 (FedAvg/FedProx, FedAvg/FedNova, FedProx/FedNova)", "Bonferroni-corrected α = 0.0167"],
  ];
  TABLE(new Table({
    width: { size: s2Total, type: WidthType.DXA },
    columnWidths: s2Cols,
    rows: [
      new TableRow({ tableHeader: true, children: s2Cols.map((w, i) => hCell(["Component", "Parameter", "Value", "Applies To"][i], w)) }),
      ...s2Data.map(r => s2Row(...r)),
    ],
  }));
  NOTE("* Grid-searched hyperparameters. DiabetesNet dropout 0.18 in layer 3 was set to reduce regularisation in the narrower final hidden layer. SCAFFOLD uses SGD rather than AdamW to satisfy the theoretical convergence guarantees of Karimireddy et al. (ICML 2020).");
  PAGE();

  // ── TABLE S3 ─────────────────────────────────────────────────────────────────
  H1("Table S3. Node Partition Details");
  P("NHANES 2013–2020 records (n = 15,650 after QC exclusions) were partitioned into three federated nodes by demographic stratification. Node sizes reflect population-proportional sampling; the three nodes together comprise 11,906 records.");
  BLANK();
  const s3Cols = [1400, 1900, 900, 800, 900, 900, 1226];
  const s3Total = s3Cols.reduce((a, b) => a + b, 0);
  function s3Row(node, desc, age, n, tr, val, prev) {
    return new TableRow({
      children: [s3Cols[0], s3Cols[1], s3Cols[2], s3Cols[3], s3Cols[4], s3Cols[5], s3Cols[6]].map((w, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 60, bottom: 60, left: 80, right: 80 },
          width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: [node, desc, age, n, tr, val, prev][i], font: FONT, size: 18 })], spacing: { line: 300, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  TABLE(new Table({
    width: { size: s3Total, type: WidthType.DXA },
    columnWidths: s3Cols,
    rows: [
      new TableRow({ tableHeader: true, children: s3Cols.map((w, i) => hCell(["Node", "Population Description", "Age Range", "Total n", "Training n (80%)", "Validation n (20%)", "Diabetes Prevalence (%)"][i], w)) }),
      s3Row("Node A", "Young Urban — community health clinic (young adults + minority groups)", "18–44 or minority", "4,500", "3,600", "900", "~8%"),
      s3Row("Node B", "Elderly Rural — critical access hospital (elderly White/Mexican-American)", "≥55 (White or Mex-Am)", "3,406", "2,725", "681", "~28%"),
      s3Row("Node C", "Mixed Metropolitan — academic medical centre (remaining mixed age/geography)", "Mixed (remainder)", "4,000", "3,200", "800", "~15%"),
      s3Row("All nodes combined", "Total federated training data", "18+", "11,906", "9,525", "2,381", "~16% (weighted avg)"),
      s3Row("Centralised (unpartitioned)", "Full NHANES after QC; used for centralised baseline", "18+", "15,650", "12,520 (80%)", "3,130 (20%)", "~16%"),
      s3Row("BRFSS (external)", "External validation only; not partitioned into nodes", "18+", "1,282,897", "N/A", "N/A", "~13.3%"),
    ],
  }));
  NOTE("Node A sampling criterion: RIDAGEYR < 45 OR minority race (RIDRETH3 ∈ {2, 4}). Node B criterion: RIDAGEYR ≥ 55 AND White or Mexican-American (RIDRETH3 ∈ {1, 3}). Node C: all remaining records after A and B assignment.");
  PAGE();

  // ── TABLE S4 ─────────────────────────────────────────────────────────────────
  H1("Table S4. Pairwise DeLong Tests Between FL Strategies (BRFSS External, n = 1,282,897)");
  P("All three pairwise comparisons between federated learning strategies were conducted on BRFSS 2020–2022 (n = 1,282,897) using the paired DeLong structural components estimator (two-sided). Bonferroni correction applied for three simultaneous comparisons (α = 0.0167).");
  BLANK();
  const s4Cols = [1800, 900, 900, 900, 900, 1000, 1626];
  const s4Total = s4Cols.reduce((a, b) => a + b, 0);
  function s4Row(comp, auc1, auc2, dauc, z, p, sig) {
    return new TableRow({
      children: [s4Cols[0], s4Cols[1], s4Cols[2], s4Cols[3], s4Cols[4], s4Cols[5], s4Cols[6]].map((w, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 60, bottom: 60, left: 80, right: 80 },
          width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: [comp, auc1, auc2, dauc, z, p, sig][i], font: FONT, size: 18 })], spacing: { line: 300, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  TABLE(new Table({
    width: { size: s4Total, type: WidthType.DXA },
    columnWidths: s4Cols,
    rows: [
      new TableRow({ tableHeader: true, children: s4Cols.map((w, i) => hCell(["Comparison", "AUC (M1)", "AUC (M2)", "ΔAUC (M1−M2)", "z-statistic", "p-value", "Bonferroni-significant?"][i], w)) }),
      s4Row("FedAvg vs. FedProx", "0.7569", "0.7518", "+0.0051", "26.99", "< 0.001", "Yes"),
      s4Row("FedAvg vs. FedNova", "0.7569", "0.7439", "+0.0130", "72.49", "< 0.001", "Yes"),
      s4Row("FedProx vs. FedNova", "0.7518", "0.7439", "+0.0079", "36.48", "< 0.001", "Yes"),
    ],
  }));
  NOTE("Method: Paired DeLong two-sided test. Dataset: BRFSS 2020–2022 (n = 1,282,897). Bonferroni-corrected significance threshold: α = 0.0167 (= 0.05 / 3). All three comparisons are statistically significant. Note: at n = 1.28 million, even a 0.005 AUC difference is statistically detectable; absolute differences should be interpreted alongside clinical significance.");
  PAGE();

  // ── TABLE S5 ─────────────────────────────────────────────────────────────────
  H1("Table S5. FedProx Post-Hoc Calibration on BRFSS External Validation");
  P("Post-hoc probability calibration was applied to the FedProx model on the BRFSS 2020–2022 external validation set (n = 1,282,897). A 20/80 calibration/test split (rng seed = 42) was used: 256,579 calibration samples and 1,026,318 test samples.");
  BLANK();
  const s5Cols = [1800, 1000, 1000, 1000, 1000, 3226];
  const s5Total = s5Cols.reduce((a, b) => a + b, 0);
  function s5Row(method, eceBefore, eceAfter, eceRed, auc, notes) {
    return new TableRow({
      children: [s5Cols[0], s5Cols[1], s5Cols[2], s5Cols[3], s5Cols[4], s5Cols[5]].map((w, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 60, bottom: 60, left: 80, right: 80 },
          width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: [method, eceBefore, eceAfter, eceRed, auc, notes][i], font: FONT, size: 18 })], spacing: { line: 300, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  TABLE(new Table({
    width: { size: s5Total, type: WidthType.DXA },
    columnWidths: s5Cols,
    rows: [
      new TableRow({ tableHeader: true, children: s5Cols.map((w, i) => hCell(["Calibration Method", "ECE (before)", "ECE (after)", "ECE Reduction", "AUC (test)", "Notes"][i], w)) }),
      s5Row("Uncalibrated", "0.319", "—", "—", "0.752", "Baseline; model overconfident"),
      s5Row("Platt scaling", "0.319", "0.016", "95.1%", "0.752", "Logistic fit on calibration set; coef=4.58, intercept=-4.41"),
      s5Row("Isotonic regression", "0.319", "0.002", "99.5%", "0.752", "Non-parametric; best ECE; risk of overfitting small cal sets"),
      s5Row("Temperature scaling (T=2.25)", "0.319", "0.311", "2.6%", "0.752", "Single-param rescaling; least reduction for FedProx"),
    ],
  }));
  NOTE("Subgroup calibration (internal NHANES validation set, n = 3,130): Young (18–39, n = 5,477): ECE before=0.045, after Platt=0.035. Elderly (≥60, n = 5,303): ECE before=0.304, after Platt=0.024. Platt scaling disproportionately benefits elderly respondents, reducing their ECE by 92% vs. 22% for young respondents.");
  PAGE();

  // ── TABLE S6 ─────────────────────────────────────────────────────────────────
  H1("Table S6. DeLong Confidence Intervals for Demographic Fairness Gaps");
  P("DeLong 95% confidence intervals (structural components estimator, Hanley-McNeil 1988) for the elderly–young AUC gap (ΔAUC = AUC_young − AUC_elderly) on the BRFSS external validation set (n = 1,282,897). Non-overlapping CIs between centralised XGBoost [0.062–0.076] and FedAvg [0.047–0.060] confirm the 21.7% gap reduction is statistically reliable.")
  BLANK();
  const s6Cols = [2000, 1000, 1600, 1700, 1726];
  const s6Total = s6Cols.reduce((a, b) => a + b, 0);
  function s6Row(model, gap, gapCI, youngAUC, elderlyAUC) {
    return new TableRow({
      children: [s6Cols[0], s6Cols[1], s6Cols[2], s6Cols[3], s6Cols[4]].map((w, i) =>
        new TableCell({
          borders: BORDERS,
          margins: { top: 60, bottom: 60, left: 80, right: 80 },
          width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: [model, gap, gapCI, youngAUC, elderlyAUC][i], font: FONT, size: 18 })], spacing: { line: 300, lineRule: LineRuleType.AUTO } })],
        })
      ),
    });
  }
  TABLE(new Table({
    width: { size: s6Total, type: WidthType.DXA },
    columnWidths: s6Cols,
    rows: [
      new TableRow({ tableHeader: true, children: s6Cols.map((w, i) => hCell(["Model", "Elderly Gap (point)", "Elderly Gap 95% CI", "Young AUC [95% CI]", "Elderly AUC [95% CI]"][i], w)) }),
      s6Row("XGBoost (Centralised)", "0.069", "[0.062–0.076]", "0.656 [0.649–0.662]", "0.587 [0.585–0.588]"),
      s6Row("DiabetesNet (Centralised)", "0.063", "[0.057–0.070]", "0.720 [0.714–0.727]", "0.657 [0.655–0.659]"),
      s6Row("FedAvg", "0.054", "[0.047–0.060]", "0.722 [0.716–0.728]", "0.669 [0.667–0.670]"),
      s6Row("FedProx (μ=0.1)", "0.066", "[0.059–0.072]", "0.727 [0.721–0.733]", "0.661 [0.660–0.663]"),
      s6Row("FedNova (τ={5,3,4})", "0.064", "[0.058–0.071]", "0.715 [0.708–0.721]", "0.650 [0.649–0.652]"),
      s6Row("SCAFFOLD (SGD, 50 rds)", "−0.092", "[−0.099–−0.085]", "0.521 [0.514–0.528]", "0.613 [0.611–0.614]"),
    ],
  }));
  NOTE("All CIs computed by DeLong structural components estimator (Hanley-McNeil 1988) on BRFSS external set (n = 1,282,897). SCAFFOLD fairness gap is negative: elderly subgroup outperforms young (attributed to SGD convergence dynamics, not a genuine fairness advantage; see manuscript Section 4.6 and 5.2).");

  return new Document({
    numbering: { config: [bulletConfig] },
    styles: { default: { document: { run: { font: FONT, size: FONT_SIZE } } } },
    sections: [{
      properties: {
        page: { size: { width: A4_W, height: A4_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } },
      },
      headers: {
        default: new Header({ children: [new Paragraph({ children: [new TextRun({ text: "Supplementary Material — Pall — FL for Diabetes Risk Prediction", font: FONT, size: 20, italics: true })], alignment: AlignmentType.RIGHT })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ children: [new TextRun({ text: "Page S", font: FONT, size: 20 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })], alignment: AlignmentType.CENTER })] }),
      },
      children,
    }],
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// WRITE ALL THREE FILES
// ══════════════════════════════════════════════════════════════════════════════

async function main() {
  const outDir = "D:/Projects/diabetes_prediction_project";

  // ── WITHOUT figures (placeholders) — for journal portal upload ──────────────
  console.log("\n[1/5] Building manuscript (no figures — journal submission)...");
  const msNoFig = buildManuscript(false);
  const msNoFigBuf = await Packer.toBuffer(msNoFig);
  fs.writeFileSync(`${outDir}/FL_Diabetes_CMPB_FINAL_SUBMISSION.docx`, msNoFigBuf);
  console.log(`  -> FL_Diabetes_CMPB_FINAL_SUBMISSION.docx  (${Math.round(msNoFigBuf.length/1024)} KB)`);

  // ── WITH figures embedded — for editor/reviewer preview ────────────────────
  console.log("[2/5] Building manuscript (figures embedded — reviewer preview)...");
  const msWithFig = buildManuscript(true);
  const msWithFigBuf = await Packer.toBuffer(msWithFig);
  fs.writeFileSync(`${outDir}/FL_Diabetes_CMPB_FINAL_with_figures.docx`, msWithFigBuf);
  console.log(`  -> FL_Diabetes_CMPB_FINAL_with_figures.docx  (${Math.round(msWithFigBuf.length/1024)} KB)`);

  // ── Cover letter ─────────────────────────────────────────────────────────────
  console.log("[3/5] Building cover letter...");
  const cl = buildCoverLetter();
  const clBuf = await Packer.toBuffer(cl);
  fs.writeFileSync(`${outDir}/Cover_Letter_CMPB_FINAL.docx`, clBuf);
  console.log(`  -> Cover_Letter_CMPB_FINAL.docx  (${Math.round(clBuf.length/1024)} KB)`);

  // ── Supplementary ─────────────────────────────────────────────────────────────
  console.log("[4/5] Building supplementary...");
  const sp = buildSupplementary();
  const spBuf = await Packer.toBuffer(sp);
  fs.writeFileSync(`${outDir}/FL_Diabetes_Supplementary_CMPB.docx`, spBuf);
  console.log(`  -> FL_Diabetes_Supplementary_CMPB.docx  (${Math.round(spBuf.length/1024)} KB)`);

  console.log("\n[5/5] Done. Output files:");
  console.log("  SUBMIT TO JOURNAL:");
  console.log("    FL_Diabetes_CMPB_FINAL_SUBMISSION.docx          (figure placeholders)");
  console.log("    Cover_Letter_CMPB_FINAL.docx");
  console.log("    FL_Diabetes_Supplementary_CMPB.docx");
  console.log("    federated/results/figures/fig1_architecture.png     (upload separately)");
  console.log("    ...fig2 through fig8.png");
  console.log("  REVIEWER PREVIEW (do NOT upload to portal):");
  console.log("    FL_Diabetes_CMPB_FINAL_with_figures.docx");
}

main().catch(e => { console.error(e); process.exit(1); });
