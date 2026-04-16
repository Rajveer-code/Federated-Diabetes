"use strict";

const fs = require("fs");
const path = require("path");

// Try to require docx from global node_modules
let docxModule;
try {
  docxModule = require("docx");
} catch (e) {
  // Try common global paths
  const globalPaths = [
    "C:\\Users\\Asus\\AppData\\Roaming\\npm\\node_modules\\docx",
    "C:\\Program Files\\nodejs\\node_modules\\docx",
    "/usr/lib/node_modules/docx",
    "/usr/local/lib/node_modules/docx",
  ];
  for (const p of globalPaths) {
    if (fs.existsSync(p)) {
      docxModule = require(p);
      break;
    }
  }
  if (!docxModule) throw new Error("Cannot find docx module: " + e.message);
}

const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  ShadingType,
  ImageRun,
  convertInchesToTwip,
  NumberFormat,
  LevelFormat,
  PageOrientation,
} = docxModule;

const outputPath = "D:\\Projects\\diabetes_prediction_project\\FL_Diabetes_Manuscript_v2.docx";
const plotDir = "D:\\Projects\\diabetes_prediction_project\\federated\\plots\\";

// ─── Spacing helpers ──────────────────────────────────────────────────────────
const BODY_SPACING = { line: 480, lineRule: "auto", before: 0, after: 120 };
const HEADING_SPACING = { before: 240, after: 120 };

function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: BODY_SPACING,
    style: opts.style || undefined,
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 24,
        bold: opts.bold || false,
        italics: opts.italics || false,
      }),
    ],
  });
}

function boldPara(text) {
  return bodyPara(text, { bold: true });
}

function headingPara(text) {
  return new Paragraph({
    spacing: HEADING_SPACING,
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 24,
        bold: true,
      }),
    ],
  });
}

function subheadingPara(text) {
  return new Paragraph({
    spacing: { before: 200, after: 80 },
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 24,
        bold: true,
        italics: false,
      }),
    ],
  });
}

function emptyPara() {
  return new Paragraph({
    spacing: { line: 480, lineRule: "auto" },
    children: [new TextRun({ text: "", font: "Times New Roman", size: 24 })],
  });
}

function captionPara(text) {
  return new Paragraph({
    spacing: { before: 80, after: 160 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 20,
        italics: true,
      }),
    ],
  });
}

function titlePara(text) {
  return new Paragraph({
    spacing: { before: 0, after: 240 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 28,
        bold: true,
      }),
    ],
  });
}

function centerPara(text, opts = {}) {
  return new Paragraph({
    spacing: BODY_SPACING,
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 24,
        bold: opts.bold || false,
        italics: opts.italics || false,
      }),
    ],
  });
}

// ─── Image helper ─────────────────────────────────────────────────────────────
function imagePara(filename, widthIn, heightIn) {
  const fullPath = plotDir + filename;
  let imgData;
  try {
    imgData = fs.readFileSync(fullPath);
  } catch (err) {
    console.warn("Warning: could not read image:", fullPath, err.message);
    return emptyPara();
  }
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    alignment: AlignmentType.CENTER,
    children: [
      new ImageRun({
        data: imgData,
        transformation: {
          width: Math.round(widthIn * 96),
          height: Math.round(heightIn * 96),
        },
        type: "png",
      }),
    ],
  });
}

// ─── Table helpers ────────────────────────────────────────────────────────────
const TABLE_BORDER = {
  top: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  left: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  right: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
};

function makeCell(text, widthDxa, isHeader = false, opts = {}) {
  return new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    shading: isHeader
      ? { type: ShadingType.CLEAR, fill: "D5E8F0", color: "auto" }
      : opts.shading
      ? { type: ShadingType.CLEAR, fill: opts.shading, color: "auto" }
      : { type: ShadingType.CLEAR, fill: "FFFFFF", color: "auto" },
    borders: TABLE_BORDER,
    children: [
      new Paragraph({
        spacing: { before: 60, after: 60 },
        alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: [
          new TextRun({
            text: text,
            font: "Times New Roman",
            size: 20,
            bold: isHeader || opts.bold || false,
          }),
        ],
      }),
    ],
  });
}

function tableTitlePara(text) {
  return new Paragraph({
    spacing: { before: 240, after: 80 },
    children: [
      new TextRun({
        text: text,
        font: "Times New Roman",
        size: 22,
        bold: true,
      }),
    ],
  });
}

// ─── TABLE 1: Internal Validation Metrics ─────────────────────────────────────
// Total width: 9360 DXA  (1-inch margins on US Letter = 6.5in * 1440 = 9360)
// Columns: Model(2200) | AUC(1300) | 95%CI(1800) | Brier(900) | F1(900) | Sens(1080) | Spec(1180)
// Sum: 2200+1300+1800+900+900+1080+1180 = 9360 ✓
function makeTable1() {
  const W = [2200, 1300, 1800, 900, 900, 1080, 1180];
  const headers = ["Model", "AUC", "95% CI", "Brier", "F1", "Sens", "Spec"];

  const rows_data = [
    ["Published XGBoost (Pall et al.)", "0.794", "N/A", "0.123", "0.518", "0.762", "0.695"],
    ["Centralised XGBoost (replicated)", "0.769", "0.760–0.777", "0.181", "0.462", "0.792", "0.626"],
    ["FedAvg (50 rounds)", "0.788", "0.779–0.796", "0.174", "0.472", "0.808", "0.631"],
    ["FedProx (μ=0.1)", "0.785", "0.776–0.793", "0.179", "0.466", "0.825", "0.608"],
    ["FedNova (τ={5,3,4})", "0.786", "0.778–0.794", "0.196", "0.473", "0.782", "0.651"],
  ];

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => makeCell(h, W[i], true, { center: true })),
  });

  const dataRows = rows_data.map(
    (row) =>
      new TableRow({
        children: row.map((cell, i) =>
          makeCell(cell, W[i], false, { center: i > 0 })
        ),
      })
  );

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: W,
    borders: TABLE_BORDER,
    rows: [headerRow, ...dataRows],
  });
}

// ─── TABLE 2: External Validation Metrics (BRFSS) ────────────────────────────
// Columns: Model(2200) | AUC(1200) | 95%CI(1760) | Brier(900) | F1(900) | Sens(1100) | Spec(1300)
// Sum: 2200+1200+1760+900+900+1100+1300 = 9360 ✓
function makeTable2() {
  const W = [2200, 1200, 1760, 900, 900, 1100, 1300];
  const headers = ["Model", "AUC", "95% CI", "Brier", "F1", "Sens", "Spec"];

  const rows_data = [
    ["Published XGBoost (Pall et al.)", "0.717", "N/A", "N/A", "N/A", "N/A", "N/A"],
    ["Centralised XGBoost (replicated)", "0.700", "N/A", "0.322", "0.318", "0.736", "0.556"],
    ["FedAvg (50 rounds)", "0.757", "0.756–0.758", "0.217", "0.355", "0.768", "0.607"],
    ["FedProx (μ=0.1)", "0.752", "0.751–0.753", "0.242", "0.353", "0.735", "0.626"],
    ["FedNova (τ={5,3,4})", "0.744", "0.743–0.745", "0.266", "0.351", "0.718", "0.635"],
  ];

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => makeCell(h, W[i], true, { center: true })),
  });

  const dataRows = rows_data.map(
    (row) =>
      new TableRow({
        children: row.map((cell, i) =>
          makeCell(cell, W[i], false, { center: i > 0 })
        ),
      })
  );

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: W,
    borders: TABLE_BORDER,
    rows: [headerRow, ...dataRows],
  });
}

// ─── TABLE 3: Fairness Summary ─────────────────────────────────────────────────
// Columns: Model(2100) | Young AUC(1380) | Elderly AUC(1380) | Gap(1100) | Gap Change(1500) | Note(1900)
// Sum: 2100+1380+1380+1100+1500+1900 = 9360 ✓
function makeTable3() {
  const W = [2100, 1380, 1380, 1100, 1500, 1900];
  const headers = [
    "Model",
    "AUC (18–39)",
    "AUC (60+)",
    "Gap",
    "Gap vs Centralised",
    "Source",
  ];

  const rows_data = [
    ["Published XGBoost (internal)", "0.742", "0.607", "0.135", "Reference (internal)", "NHANES (internal)"],
    ["Centralised XGBoost (ext.)", "0.656", "0.587", "0.069", "Baseline", "BRFSS external"],
    ["FedAvg (ext.)", "0.722", "0.669", "0.054", "-21.7%", "BRFSS external"],
    ["FedProx (ext.)", "0.727", "0.661", "0.066", "-4.3%", "BRFSS external"],
    ["FedNova (ext.)", "0.715", "0.650", "0.064", "-7.2%", "BRFSS external"],
  ];

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => makeCell(h, W[i], true, { center: true })),
  });

  const dataRows = rows_data.map(
    (row) =>
      new TableRow({
        children: row.map((cell, i) =>
          makeCell(cell, W[i], false, { center: i > 0 })
        ),
      })
  );

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: W,
    borders: TABLE_BORDER,
    rows: [headerRow, ...dataRows],
  });
}

// ─── Mixed-run paragraph helper (inline bold + normal) ────────────────────────
function mixedPara(runs, opts = {}) {
  return new Paragraph({
    spacing: BODY_SPACING,
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: runs.map(
      (r) =>
        new TextRun({
          text: r.text,
          font: "Times New Roman",
          size: 24,
          bold: r.bold || false,
          italics: r.italics || false,
        })
    ),
  });
}

// ─── Numbered list paragraph ──────────────────────────────────────────────────
function numberedPara(num, text) {
  return new Paragraph({
    spacing: { line: 480, lineRule: "auto", before: 60, after: 60 },
    indent: { left: 720, hanging: 360 },
    children: [
      new TextRun({
        text: num + ". " + text,
        font: "Times New Roman",
        size: 24,
      }),
    ],
  });
}

function referencePara(num, text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 720, hanging: 360 },
    children: [
      new TextRun({
        text: "[" + num + "] " + text,
        font: "Times New Roman",
        size: 22,
      }),
    ],
  });
}

// ─── Build document sections ──────────────────────────────────────────────────
const children = [];

// ════════════════════════════════════════════════════════
// TITLE PAGE
// ════════════════════════════════════════════════════════
children.push(emptyPara());
children.push(
  titlePara(
    "Federated Learning for Privacy-Preserving Diabetes Risk Prediction Across Demographically Heterogeneous Hospital Nodes: External Validation with 1.28 Million Records"
  )
);
children.push(emptyPara());
children.push(centerPara("Rajveer Singh Pall\u00B9, Sameer Yadav\u00B9, [et al.]"));
children.push(emptyPara());
children.push(
  centerPara(
    "\u00B9Department of Computer Science and Engineering, Gyan Ganga Institute of Technology and Sciences, Jabalpur, Madhya Pradesh, India"
  )
);
children.push(emptyPara());
children.push(
  centerPara(
    "Manuscript submitted to Journal of Biomedical Informatics (JBI), Elsevier"
  )
);
children.push(emptyPara());
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// HIGHLIGHTS
// ════════════════════════════════════════════════════════
children.push(headingPara("Highlights"));
children.push(
  numberedPara(
    1,
    "Federated learning trains across three demographically heterogeneous hospital nodes without sharing raw patient data"
  )
);
children.push(
  numberedPara(
    2,
    "FedAvg achieves external AUC = 0.757 (95% CI: 0.756-0.758) versus published baseline of 0.717 (+5.7 percentage points)"
  )
);
children.push(
  numberedPara(
    3,
    "FedAvg reduces the external elderly fairness gap from 0.069 to 0.054 (-21.7%) on 1.28 million BRFSS records"
  )
);
children.push(
  numberedPara(
    4,
    "Differential privacy with epsilon <= 5 causes model collapse (AUC approx. 0.5) at this training scale, demonstrating fundamental privacy-utility tension"
  )
);
children.push(
  numberedPara(
    5,
    "DeLong and stratified bootstrap 95% confidence intervals confirm statistical robustness of all reported AUC values"
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// ABSTRACT
// ════════════════════════════════════════════════════════
children.push(headingPara("Abstract"));
children.push(
  mixedPara([{ text: "Background: ", bold: true }, { text: "Type 2 diabetes affects over 537 million adults worldwide and disproportionately impacts elderly and minority populations. Machine learning models trained on centralised clinical datasets achieve high internal accuracy but exhibit substantial performance degradation on external populations\u2014a critical limitation for clinical deployment. Furthermore, privacy regulations such as HIPAA and GDPR prohibit sharing raw patient records across institutions." }])
);
children.push(
  mixedPara([{ text: "Objectives: ", bold: true }, { text: "We investigated whether federated learning (FL) training across three demographically heterogeneous hospital nodes improves generalisation and fairness for diabetes risk prediction without raw data sharing." }])
);
children.push(
  mixedPara([{ text: "Methods: ", bold: true }, { text: "NHANES 2015-2020 data (n=15,650) were partitioned into three federated nodes simulating demographically distinct hospital settings: a young-urban community clinic (Node A, mean age 40.8, diabetes prevalence 13.8%), an elderly-rural critical access hospital (Node B, mean age 69.1, diabetes prevalence 28.5%), and a mixed-metropolitan academic centre (Node C, mean age 45.0, diabetes prevalence 16.7%). We trained and compared three FL aggregation strategies\u2014FedAvg, FedProx (mu=0.1), and FedNova (heterogeneous local epochs per Wang et al., 2020)\u2014over 50 communication rounds. All models were externally validated on BRFSS 2020-2022 (n=1,282,897). Statistical inference used DeLong 95% confidence intervals for external AUC (O(n log n) implementation to handle 1.28M records) and stratified bootstrap confidence intervals for internal AUC (N=2,000). Differential privacy was evaluated via DP-SGD (Opacus) at epsilon in {0.5, 1.0, 2.0, 5.0, infinity}." }])
);
children.push(
  mixedPara([{ text: "Results: ", bold: true }, { text: "All FL models exceeded the published external AUC of 0.717: FedAvg achieved 0.757 (95% CI: 0.756-0.758, +5.7 pp), FedProx 0.752 (95% CI: 0.751-0.753, +4.9 pp), and FedNova 0.744 (95% CI: 0.743-0.745, +3.8 pp). FedAvg surpassed the centralised XGBoost replication on external data (0.757 vs 0.700). On internal validation, all FL models (FedAvg: 0.788, 95% CI: 0.779-0.796) outperformed the centralised XGBoost replication (0.769, 95% CI: 0.760-0.777). FedAvg reduced the external elderly fairness gap from 0.069 (centralised) to 0.054 (-21.7%). Paired DeLong testing confirmed FedAvg's advantage over FedProx (DELTA-AUC=+0.005, z=26.99, p<0.001) as statistically significant but clinically modest. DP-SGD with epsilon <= 5 collapsed to AUC approx. 0.5 at this training scale." }])
);
children.push(
  mixedPara([{ text: "Conclusions: ", bold: true }, { text: "Federated learning across demographically heterogeneous nodes substantially improves external generalisation and fairness for diabetes risk prediction. The fundamental privacy-utility tension identified at epsilon <= 5 motivates future work on larger federated cohorts and secure aggregation protocols." }])
);
children.push(emptyPara());
children.push(
  bodyPara(
    "Keywords: federated learning; diabetes prediction; external validation; fairness; differential privacy; NHANES; BRFSS; TRIPOD-AI"
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 1. INTRODUCTION
// ════════════════════════════════════════════════════════
children.push(headingPara("1. Introduction"));

children.push(
  bodyPara(
    "Type 2 diabetes mellitus (T2DM) represents one of the most pressing global health challenges, affecting an estimated 537 million adults in 2021 and projected to reach 783 million by 2045 [1]. Early and accurate risk stratification is essential to enabling preventative interventions, yet clinical deployment of machine learning (ML) prediction models is hindered by two fundamental problems: poor generalisation to external populations and disparate performance across demographic subgroups."
  )
);
children.push(
  bodyPara(
    "The premise of this work builds directly on a recently published IEEE paper that trained a centralised XGBoost model on NHANES 2015-2020 data and achieved an internal AUC of 0.794\u2014commendable performance that, however, degraded substantially to 0.717 on external validation against BRFSS 2020-2022 [2]. More critically, that centralised model exhibited a severe fairness gap: elderly patients (age >=60) achieved AUC=0.607 versus 0.742 for young adults (18-39), a gap of 0.135. This performance disparity is not a modelling artefact\u2014it reflects genuine distributional heterogeneity between age groups that is exacerbated when training data are drawn from a single, centralised pool."
  )
);
children.push(
  bodyPara(
    "Federated learning (FL) offers a principled solution to both problems simultaneously. By training across multiple geographically and demographically distinct institutions without centralising raw patient records, FL can leverage population diversity as a training signal rather than a confounder [3]. FL also directly addresses regulatory barriers: HIPAA's minimum necessary standard and GDPR's data minimisation principle both prohibit sharing identifiable clinical records across institutional boundaries\u2014barriers that FL's weight-sharing protocol circumvents entirely."
  )
);
children.push(bodyPara("This paper makes the following contributions:"));
children.push(emptyPara());
children.push(
  numberedPara(
    1,
    "(RQ1) Accuracy: Do federated models trained on demographically heterogeneous nodes match or exceed the published external AUC of 0.717?"
  )
);
children.push(
  numberedPara(
    2,
    "(RQ2) Fairness: Does federated training reduce the elderly performance gap from 0.135, specifically by leveraging Node B (elderly-rural hospital, 82.4% elderly patients)?"
  )
);
children.push(
  numberedPara(
    3,
    "(RQ3) Aggregation Strategy: Which FL strategy\u2014FedAvg, FedProx, or FedNova\u2014performs best on non-IID clinical data?"
  )
);
children.push(
  numberedPara(
    4,
    "(RQ4) Privacy: How does epsilon-differential privacy (via DP-SGD) affect the accuracy-fairness trade-off at clinical FL dataset scales?"
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 2. RELATED WORK
// ════════════════════════════════════════════════════════
children.push(headingPara("2. Related Work"));

children.push(subheadingPara("2.1 Federated Learning in Healthcare"));
children.push(
  bodyPara(
    "McMahan et al. (2017) introduced FedAvg and demonstrated convergence guarantees under i.i.d. data conditions [3]. Healthcare applications have since demonstrated FL's utility in radiology [9], electronic health records [13], and genomics [14]. However, non-IID data distributions\u2014the rule rather than the exception in clinical settings\u2014degrade FedAvg's performance due to client drift [15]."
  )
);

children.push(subheadingPara("2.2 Fairness in Clinical Prediction Models"));
children.push(
  bodyPara(
    "Obermeyer et al. (2019) demonstrated systematic racial bias in commercial healthcare algorithms [10]. For diabetes specifically, Gianfrancesco et al. (2018) documented performance disparities across race, age, and socioeconomic status. The TRIPOD-AI reporting guidelines require explicit fairness evaluation across demographic subgroups [11]."
  )
);

children.push(subheadingPara("2.3 Federated Aggregation Strategies"));
children.push(
  bodyPara(
    "FedProx (Li et al., 2020) adds a proximal term (mu||w-w_global||^2) to each client's local objective, bounding divergence from the global model and improving convergence on heterogeneous data [4]. FedNova (Wang et al., 2020) addresses objective inconsistency in non-IID settings through normalised averaging that accounts for heterogeneous local training steps [5]."
  )
);

children.push(subheadingPara("2.4 Differential Privacy in Federated Learning"));
children.push(
  bodyPara(
    "Abadi et al. (2016) introduced DP-SGD for training neural networks with formal (epsilon, delta)-differential privacy guarantees [7]. Bagdasaryan et al. (2019) showed that DP-SGD disproportionately degrades accuracy for underrepresented subgroups, raising fairness concerns at tight epsilon budgets [8]."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 3. METHODS
// ════════════════════════════════════════════════════════
children.push(headingPara("3. Methods"));

children.push(subheadingPara("3.1 Datasets"));
children.push(
  mixedPara([
    { text: "Internal (NHANES 2015-2020): ", bold: true, italics: false },
    { text: "We utilised the National Health and Nutrition Examination Survey 2015-2020 cycles (n=15,650 adults >=18 years with complete case data). Eight features matching the IEEE paper [2] were used: age, sex, race/ethnicity, BMI, smoking status, physical activity, history of heart attack, and history of stroke. Diabetes outcome (HbA1c >=6.5% or fasting glucose >=126 mg/dL, or self-reported physician diagnosis) yielded a prevalence of 18.6% (n_pos=2,908)." },
  ])
);
children.push(
  mixedPara([
    { text: "External (BRFSS 2020-2022): ", bold: true, italics: false },
    { text: "The Behavioral Risk Factor Surveillance System (n=1,282,897, prevalence 13.3%) provided an independent external validation cohort with equivalent feature mapping. Features were standardised using a global scaler fitted exclusively on NHANES training data to prevent preprocessing domain mismatch." },
  ])
);

children.push(subheadingPara("3.2 Federated Node Design"));
children.push(
  bodyPara(
    "Three nodes were constructed from NHANES data to simulate demographically distinct hospital settings:"
  )
);
children.push(
  numberedPara(
    1,
    "Node A (Young Urban, n approx. 4,500): Community health clinic serving a minority-heavy population. Mean age 40.8 years, diabetes prevalence 13.8%."
  )
);
children.push(
  numberedPara(
    2,
    "Node B (Elderly Rural, n approx. 3,400): Rural critical access hospital. Mean age 69.1 years, diabetes prevalence 28.5%, 82.4% elderly patients. This node directly targets the population on which the IEEE paper performed worst (AUC=0.607)."
  )
);
children.push(
  numberedPara(
    3,
    "Node C (Mixed Metro, n approx. 4,000): Academic medical centre with a mixed-age population. Mean age 45.0 years, diabetes prevalence 16.7%."
  )
);

children.push(subheadingPara("3.3 Neural Network Architecture"));
children.push(
  bodyPara(
    "A three-layer feedforward network (DiabetesNet) was implemented in PyTorch: Input(8) -> Dense(64, BatchNorm, ReLU, Dropout(0.3)) -> Dense(32, BatchNorm, ReLU, Dropout(0.3)) -> Dense(16, BatchNorm, ReLU, Dropout(0.18)) -> Dense(1). Weights were initialised with Kaiming uniform initialisation. Training used BCEWithLogitsLoss with class-weight adjustment. Optimiser: AdamW (lr=0.001, weight_decay=1e-4). AMP mixed-precision training (FP16) was used for GPU efficiency."
  )
);

children.push(subheadingPara("3.4 Federated Aggregation Strategies"));
children.push(
  mixedPara([
    { text: "FedAvg ", bold: true },
    { text: "[McMahan et al., 2017]: Server aggregates client weights by weighted average (weights proportional to local dataset size) after 50 communication rounds with 5 local epochs per round." },
  ])
);
children.push(
  mixedPara([
    { text: "FedProx ", bold: true },
    { text: "(mu=0.1) [Li et al., 2020]: Adds proximal regularisation term (mu/2)||w - w_global||^2 to each client's loss. Bounded client divergence stabilises convergence on non-IID Node B data." },
  ])
);
children.push(
  mixedPara([
    { text: "FedNova ", bold: true },
    { text: "(tau_A=5, tau_B=3, tau_C=4) [Wang et al., 2020]: Assigns heterogeneous local epochs inversely proportional to distribution shift. Node B (highest shift, elderly-rural) receives tau_B=3 (fewest local steps); Node A (lowest shift, young-urban) receives tau_A=5 (most). Normalised averaging corrects for the resulting objective inconsistency. Note: equal tau for all nodes collapses FedNova algebraically to FedAvg." },
  ])
);

children.push(subheadingPara("3.5 Differential Privacy"));
children.push(
  bodyPara(
    "DP-SGD was implemented using the Opacus library (v1.x) with max gradient norm clipping of 1.0. Privacy budgets epsilon in {0.5, 1.0, 2.0, 5.0} were evaluated at delta=1e-5, with a no-DP condition (epsilon=infinity) as control. The ReLU activation in DiabetesNet required inplace=False to be compatible with Opacus per-sample gradient hooks."
  )
);

children.push(subheadingPara("3.6 Statistical Analysis"));
children.push(
  mixedPara([
    { text: "Internal AUC: ", bold: true },
    { text: "Stratified bootstrap 95% confidence intervals (N=2,000 resamples, percentile method) ensuring positive and negative cases are resampled separately." },
  ])
);
children.push(
  mixedPara([
    { text: "External AUC: ", bold: true },
    { text: "DeLong structural-component confidence intervals [DeLong et al., 1988] using an O(n log n) searchsorted implementation to handle 1.28M records (the standard kernel-matrix approach would require approx. 708 GB RAM at this scale)." },
  ])
);
children.push(
  mixedPara([
    { text: "Paired comparisons: ", bold: true },
    { text: "Paired DeLong test with covariance-corrected variance estimation for FedAvg vs FedProx on the same external test set." },
  ])
);
children.push(bodyPara("All analyses used alpha=0.05."));

children.push(subheadingPara("3.7 Fairness Metrics"));
children.push(
  bodyPara(
    "Three fairness metrics were computed: (1) subgroup AUC by age group (18-39, 40-59, >=60), BMI category (Normal/Overweight/Obese), and sex; (2) elderly fairness gap (AUC_young - AUC_elderly); (3) Equalized Odds Difference (EOD = max(|DELTA-TPR|, |DELTA-FPR|) across age groups); (4) Youden's J index per subgroup at the global Youden's J threshold."
  )
);

children.push(subheadingPara("3.8 Compliance"));
children.push(
  bodyPara(
    "Reporting followed TRIPOD-AI guidelines. Both NHANES and BRFSS are publicly available deidentified datasets; no IRB approval was required."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 4. RESULTS
// ════════════════════════════════════════════════════════
children.push(headingPara("4. Results"));

children.push(subheadingPara("4.1 Internal Validation (NHANES)"));
children.push(
  bodyPara(
    "Table 1 presents internal performance metrics. All FL models exceeded the centralised XGBoost replication on AUC: FedAvg achieved 0.788 (95% CI: 0.779-0.796), FedProx 0.785 (95% CI: 0.776-0.793), and FedNova 0.786 (95% CI: 0.778-0.794), compared to 0.769 (95% CI: 0.760-0.777) for centralised XGBoost. FL models did not attain the published AUC of 0.794, consistent with the expectation that a neural network trained on smaller per-node datasets yields lower internal accuracy than the published model trained on the full centralised NHANES cohort. Figure 3 shows federated convergence over 50 rounds; all strategies converge within 25 rounds."
  )
);
children.push(emptyPara());
children.push(tableTitlePara("Table 1. Internal Validation Performance Metrics (NHANES 2015-2020, n=15,650, n_pos=2,908, prevalence=18.6%)"));
children.push(makeTable1());
children.push(emptyPara());

// Figure 1
children.push(imagePara("01_centralised_roc.png", 5.25, 4.5));
children.push(
  captionPara(
    "Figure 1. ROC curve of centralised XGBoost baseline (5-fold CV on NHANES 2015-2020). AUC=0.769 (95% CI: 0.760-0.777)."
  )
);
children.push(emptyPara());

// Figure 3
children.push(imagePara("03_fl_convergence.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 3. Federated learning convergence over 50 communication rounds. FedAvg, FedProx, and FedNova all converge within 25 rounds."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.2 External Validation (BRFSS 2020-2022)"));
children.push(
  bodyPara(
    "Table 2 presents external performance on BRFSS (n=1,282,897). All three FL models substantially exceeded the published external AUC of 0.717: FedAvg 0.757 (95% CI: 0.756-0.758, +5.7 pp), FedProx 0.752 (95% CI: 0.751-0.753, +4.9 pp), FedNova 0.744 (95% CI: 0.743-0.745, +3.8 pp). All FL models also exceeded the centralised XGBoost external AUC (0.700). The consistent improvement across all three strategies confirms that FL training on demographically heterogeneous nodes substantially improves generalisation. Figure 10 displays external ROC curves."
  )
);
children.push(emptyPara());
children.push(tableTitlePara("Table 2. External Validation Performance Metrics (BRFSS 2020-2022, n=1,282,897, prevalence=13.3%)"));
children.push(makeTable2());
children.push(emptyPara());

// Figure 10
children.push(imagePara("10_external_validation_roc.png", 6.0, 5.25));
children.push(
  captionPara(
    "Figure 10. External validation ROC curves on BRFSS 2020-2022 (n=1,282,897). FedAvg achieves AUC=0.757 vs published 0.717."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.3 Federated Strategy Comparison"));
children.push(
  bodyPara(
    "FedAvg achieved the highest external AUC (0.757), outperforming FedProx (0.752) and FedNova (0.744). The paired DeLong test confirmed that FedAvg's advantage over FedProx is statistically significant (DELTA-AUC=+0.005, z=26.99, p<0.001), although the absolute difference is clinically modest (0.5 percentage points). The DeLong p-value attains extreme significance largely due to the massive sample size (n=1.28M); effect size interpretation should take precedence over p-values in this context. Figure 4 summarises strategy comparison."
  )
);
children.push(emptyPara());

// Figure 4
children.push(imagePara("04_fl_strategy_comparison.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 4. Final-round AUC comparison of three aggregation strategies. FedAvg achieves highest external AUC (0.757)."
  )
);
children.push(emptyPara());

// Figure 12
children.push(imagePara("12_fednova_corrected.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 12. FedNova corrected convergence with heterogeneous tau={5,3,4} per Wang et al. (2020)."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.4 Fairness Analysis"));
children.push(
  bodyPara(
    "Table 3 presents the external fairness analysis. FedAvg achieved the lowest external elderly gap (0.054), representing a 21.7% reduction from the centralised baseline (0.069) and a 60% reduction from the published internal gap (0.135). FedProx (gap=0.066) and FedNova (gap=0.064) also reduced the gap relative to the centralised baseline. On external absolute performance, FedAvg improved elderly AUC from 0.587 (centralised) to 0.669 (+8.2 pp) and young adult AUC from 0.656 to 0.722 (+6.6 pp)\u2014a proportionally greater benefit for the elderly subgroup. Internal Equalized Odds Difference was 0.552 for centralised XGBoost and 0.713 for FedProx at the global Youden's J threshold; this counterintuitive EOD increase reflects the federated model's enhanced sensitivity for elderly patients (TPR=0.969) at the cost of elevated false-positive rate (FPR=0.907) under the global threshold. Figures 6, 7, and 11 present detailed fairness profiles."
  )
);
children.push(emptyPara());
children.push(tableTitlePara("Table 3. Fairness Summary: Age-Subgroup AUC and Elderly Fairness Gap (External BRFSS Validation)"));
children.push(makeTable3());
children.push(emptyPara());

// Figure 2
children.push(imagePara("02_centralised_fairness_age.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 2. Age-subgroup AUC of centralised XGBoost vs published IEEE model. The elderly fairness gap (0.046) is the primary problem federated learning addresses."
  )
);
children.push(emptyPara());

// Figure 6
children.push(imagePara("06_fairness_age_comparison.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 6. Age-subgroup AUC comparison: published, centralised, and FedProx (internal NHANES)."
  )
);
children.push(emptyPara());

// Figure 7
children.push(imagePara("07_fairness_full_profile.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 7. Full fairness profile across age, BMI, and sex subgroups for centralised vs federated models."
  )
);
children.push(emptyPara());

// Figure 11
children.push(imagePara("11_external_validation_fairness.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 11. External fairness comparison: age-subgroup AUC on BRFSS for all models. FedAvg reduces elderly gap from 0.069 to 0.054 (-21.7%)."
  )
);
children.push(emptyPara());

// Figure 8
children.push(imagePara("08_node_b_elderly_analysis.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 8. Node B (Elderly Rural) deep-dive analysis. Node B has 82.4% elderly patients and 28.5% diabetes prevalence."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.5 Differential Privacy"));
children.push(
  bodyPara(
    "Figure 5 shows the privacy-utility trade-off. DP-SGD with epsilon <= 5 produced model collapse (AUC approx. 0.50) across all evaluated budgets, while the no-DP condition achieved AUC=0.766. This collapse is consistent with Bagdasaryan et al. (2019) [8]: with only approx. 3,000-4,500 samples per node, per-sample gradient clipping severely restricts the signal-to-noise ratio. Viable DP protection would require substantially larger federated cohorts (n >= 50,000 per node) or secure aggregation protocols that enable tighter privacy accounting."
  )
);
children.push(emptyPara());

// Figure 5
children.push(imagePara("05_dp_tradeoff.png", 6.0, 3.59));
children.push(
  captionPara(
    "Figure 5. Privacy-utility trade-off for DP-SGD. All epsilon <= 5 cause model collapse (AUC approx. 0.5) at this dataset scale; no-DP baseline AUC=0.766."
  )
);
children.push(emptyPara());

// Figure 9 (publication summary)
children.push(imagePara("09_publication_summary_2x2.png", 6.0, 6.0));
children.push(
  captionPara(
    "Figure 9. 2x2 publication summary figure combining convergence, strategy comparison, fairness, and DP results."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 5. DISCUSSION
// ════════════════════════════════════════════════════════
children.push(headingPara("5. Discussion"));

children.push(subheadingPara("5.1 Key Finding: FL Improves External Generalisation"));
children.push(
  bodyPara(
    "The most important finding is that all three FL strategies substantially outperform both the centralised replication and the published model on external validation\u2014not just by a small margin, but by 3.8-5.7 percentage points in AUC on 1.28 million records. This is a meaningful effect for a screening tool where a 1-pp AUC improvement translates to thousands of correctly classified cases at scale."
  )
);
children.push(
  bodyPara(
    "The mechanistic explanation is that FL training on heterogeneous nodes acts as a form of implicit domain augmentation. Each node presents a different age, sex, and comorbidity distribution; by aggregating gradients from all three, the global model learns a more robust feature representation that transfers better to unseen populations. This generalisation benefit of population-diverse FL is consistent with prior findings in imaging-based FL [9]."
  )
);

children.push(subheadingPara("5.2 Fairness: Why FedAvg Wins"));
children.push(
  bodyPara(
    "FedAvg's superior external fairness (gap=0.054) compared to FedProx (0.066) and FedNova (0.064) is somewhat counterintuitive\u2014FedProx was specifically designed for heterogeneous settings and FedNova for non-IID distributions. We hypothesise that the proximal regularisation in FedProx, while preventing client drift, also dampens Node B's unique elderly-specific learning signal, resulting in a global model that is slightly less tailored to elderly patients. FedNova's normalisation may similarly attenuate the disproportionate influence of Node B's high-prevalence (28.5%) elderly population on the aggregated gradient."
  )
);
children.push(
  bodyPara(
    "The persistence of a non-zero elderly gap (0.054) in the best FL model warrants honest interpretation. High diabetes prevalence in elderly patients (28.5% in Node B vs 13.8% in Node A) compresses the discriminative range for the elderly subgroup, creating an intrinsic floor on subgroup AUC that is not a modelling failure but a statistical consequence of prevalence asymmetry. Node-specific threshold calibration\u2014outside the scope of this work\u2014would be a natural next step."
  )
);

children.push(subheadingPara("5.3 The EOD Paradox"));
children.push(
  bodyPara(
    "The internal EOD for FedProx (0.713) exceeds that of centralised XGBoost (0.552), which might be misread as FL making fairness worse. This is a threshold artefact: at the global Youden's J threshold, FedProx achieves dramatically higher elderly sensitivity (TPR=0.969) but also higher false-positive rate (FPR=0.907). Under a subgroup-specific threshold\u2014optimal for each group independently\u2014this disparity would be substantially reduced. Reporting EOD at a global threshold without this caveat risks misleading clinical interpretation; we recommend subgroup-specific calibration for deployment."
  )
);

children.push(subheadingPara("5.4 DP-SGD: Honest Assessment"));
children.push(
  bodyPara(
    "The complete model collapse at epsilon <= 5 is an honest negative result that the field needs to see reported. With approx. 3,000-4,500 samples per node, per-sample gradient clipping destroys the learning signal under tight epsilon budgets. This is not a failure of DP-SGD in principle, but a reflection of the sample size regime. Extrapolating from the no-DP AUC (0.766) and published results showing viable DP-FL at epsilon=8-10 with n~50,000, we estimate a minimum of approx. 40,000 samples per node would be required for this architecture to sustain AUC >= 0.70 at epsilon <= 5."
  )
);

children.push(subheadingPara("5.5 Limitations"));
children.push(
  bodyPara(
    "Several limitations should be noted. First, the three hospital nodes are simulated from NHANES data; real-world FL would face additional heterogeneity from differing data collection protocols, EHR coding practices, and label definitions. Second, the NHANES-BRFSS feature mapping required encoding assumptions for smoking status and physical activity that introduce measurement invariance risk. Third, only eight features were used, excluding potentially important biomarkers (HbA1c, fasting glucose) that may not be universally available across federated nodes. Fourth, FL simulations were conducted in a single-process environment rather than true distributed infrastructure, which excludes communication overhead from runtime estimates."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 6. CONCLUSION
// ════════════════════════════════════════════════════════
children.push(headingPara("6. Conclusion"));
children.push(
  bodyPara(
    "We demonstrated that federated learning across three demographically heterogeneous simulated hospital nodes yields diabetes prediction models that substantially exceed the published external AUC (0.717) of a centralised XGBoost baseline, with FedAvg achieving 0.757 (95% CI: 0.756-0.758) on 1.28 million BRFSS records. FedAvg reduced the external elderly fairness gap by 21.7% (from 0.069 to 0.054) without sharing any raw patient data. The fundamental privacy-utility tension\u2014model collapse at epsilon <= 5 with current dataset sizes\u2014motivates future work combining larger federated cohorts, differential privacy accounting improvements, and node-specific threshold calibration. These findings support the clinical deployment of FL-based diabetes screening in systems where data sharing is legally or logistically infeasible."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// DATA AVAILABILITY
// ════════════════════════════════════════════════════════
children.push(headingPara("Data Availability"));
children.push(
  bodyPara(
    "NHANES 2015-2020 data are publicly available at https://www.cdc.gov/nchs/nhanes/. BRFSS 2020-2022 data are publicly available at https://www.cdc.gov/brfss/. No new primary data were collected."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// CODE AVAILABILITY
// ════════════════════════════════════════════════════════
children.push(headingPara("Code Availability"));
children.push(
  bodyPara(
    "All federated learning code, preprocessing scripts, and analysis notebooks are publicly available at https://github.com/rajveersinghpall/federated-diabetes-prediction under the MIT license."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// ETHICS STATEMENT
// ════════════════════════════════════════════════════════
children.push(headingPara("Ethics Statement"));
children.push(
  bodyPara(
    "Both datasets (NHANES and BRFSS) are deidentified, publicly available secondary datasets. Institutional Review Board (IRB) approval was not required under 45 CFR 46.101(b)(4) (research on publicly available data)."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// CREDIT AUTHORSHIP
// ════════════════════════════════════════════════════════
children.push(headingPara("CRediT Author Contribution Statement"));
children.push(
  bodyPara(
    "Rajveer Singh Pall: Conceptualisation, Methodology, Software, Formal Analysis, Writing - Original Draft, Writing - Review and Editing."
  )
);
children.push(
  bodyPara(
    "Sameer Yadav: Supervision, Writing - Review and Editing, Funding Acquisition."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// DECLARATION OF COMPETING INTEREST
// ════════════════════════════════════════════════════════
children.push(headingPara("Declaration of Competing Interest"));
children.push(bodyPara("The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."));
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// REFERENCES
// ════════════════════════════════════════════════════════
children.push(headingPara("References"));

const refs = [
  "IDF Diabetes Atlas, 10th edn. International Diabetes Federation, Brussels, Belgium, 2021. Available at: https://www.diabetesatlas.org",
  "Pall, R.S., Yadav, S., et al. Diabetes Risk Prediction Using XGBoost on NHANES 2015-2020: A Machine Learning Approach. IEEE [Journal], 2025.",
  "McMahan, H.B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A. Communication-Efficient Learning of Deep Networks from Decentralized Data. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017; 54: 1273-1282.",
  "Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V. Federated Optimization in Heterogeneous Networks. International Conference on Learning Representations (ICLR), 2020.",
  "Wang, J., Liu, Q., Liang, H., Joshi, G., Poor, H.V. Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization. Advances in Neural Information Processing Systems (NeurIPS), 2020; 33: 7611-7623.",
  "DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. Biometrics, 1988; 44(3): 837-845.",
  "Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., Zhang, L. Deep Learning with Differential Privacy. Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), 2016; 308-318.",
  "Bagdasaryan, E., Poursaeed, O., Shmatikov, V. Differential Privacy Has Disparate Impact on Model Accuracy. Advances in Neural Information Processing Systems (NeurIPS), 2019; 32.",
  "Rieke, N., Hancox, J., Li, W., et al. The Future of Digital Health with Federated Learning. npj Digital Medicine, 2020; 3(1): 119.",
  "Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations. Science, 2019; 366(6464): 447-453.",
  "Collins, G.S., Moons, K.G.M., Dhiman, P., et al. TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models that Use Regression or Machine Learning Methods. BMJ, 2024; 385: e078378.",
  "Hanley, J.A., McNeil, B.J. The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve. Radiology, 1982; 143(1): 29-36.",
  "Pfohl, S.R., Duan, T., Ding, D.Y., Shah, N.H. An Empirical Characterization of Fair Machine Learning for Clinical Risk Prediction. Journal of Biomedical Informatics, 2019; 113: 103621.",
  "Warnat-Herresthal, S., Schultze, H., Shastry, K.L., et al. Swarm Learning for Decentralized and Confidential Clinical Machine Learning. Nature, 2021; 594(7862): 265-270.",
  "Karimireddy, S.P., Kale, S., Mohri, M., Reddi, S.J., Stich, S.U., Suresh, A.T. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. International Conference on Machine Learning (ICML), 2020; 119: 5132-5143.",
  "Centers for Disease Control and Prevention. National Health and Nutrition Examination Survey (NHANES) 2015-2020. Available at: https://www.cdc.gov/nchs/nhanes/ [Accessed 2024].",
  "Centers for Disease Control and Prevention. Behavioral Risk Factor Surveillance System (BRFSS) 2020-2022. Available at: https://www.cdc.gov/brfss/ [Accessed 2024].",
  "Konecny, J., McMahan, H.B., Ramage, D., Richtarik, P. Federated Optimization: Distributed Machine Learning for Mobile Devices. arXiv preprint arXiv:1610.02527, 2016.",
  "Yang, Q., Liu, Y., Chen, T., Tong, Y. Federated Machine Learning: Concept and Applications. ACM Transactions on Intelligent Systems and Technology, 2019; 10(2): 1-19.",
  "Dwork, C., Roth, A. The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science, 2014; 9(3-4): 211-407.",
  "Geyer, R.C., Klein, T., Nabi, M. Differentially Private Federated Learning: A Client Level Perspective. arXiv preprint arXiv:1712.07557, 2017.",
  "Li, T., Hu, S., Beirami, A., Smith, V. Ditto: Fair and Robust Federated Learning through Personalization. International Conference on Machine Learning (ICML), 2021; 139: 6357-6368.",
  "Zhang, C., Xie, Y., Bai, H., Yu, B., Li, W., Gao, Y. A Survey on Federated Learning. Knowledge-Based Systems, 2021; 216: 106775.",
  "Wiens, J., Saria, S., Sendak, M., et al. Do No Harm: A Roadmap for Responsible Machine Learning for Health Care. Nature Medicine, 2019; 25(9): 1337-1340.",
  "Papadimitriou, G. The International Classification of Diseases (ICD-10-CM). Journal of Hospital Librarianship, 2016; 16(2): 163-165.",
  "Price, W.N., Cohen, I.G. Privacy in the Age of Medical Big Data. Nature Medicine, 2019; 25(1): 37-43.",
  "Rajpurkar, P., Chen, E., Banerjee, O., Topol, E.J. AI in Health and Medicine. Nature Medicine, 2022; 28(1): 31-38.",
  "Lundberg, S.M., Lee, S.I. A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS), 2017; 30.",
  "Breiman, L. Random Forests. Machine Learning, 2001; 45(1): 5-32.",
  "Chen, T., Guestrin, C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016; 785-794.",
  "Gianfrancesco, M.A., Tamang, S., Yazdany, J., Schmajuk, G. Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data. JAMA Internal Medicine, 2018; 178(11): 1544-1547.",
  "Steyerberg, E.W., Vickers, A.J., Cook, N.R., et al. Assessing the Performance of Prediction Models: A Framework for Traditional and Novel Measures. Epidemiology, 2010; 21(1): 128-138.",
  "Pencina, M.J., D'Agostino Sr, R.B., D'Agostino Jr, R.B., Vasan, R.S. Evaluating the Added Predictive Ability of a New Marker: From Area under the ROC Curve to Reclassification and Beyond. Statistics in Medicine, 2008; 27(2): 157-172.",
  "Youden, W.J. Index for Rating Diagnostic Tests. Cancer, 1950; 3(1): 32-35.",
  "Cook, N.R. Use and Misuse of the Receiver Operating Characteristic Curve in Risk Prediction. Circulation, 2007; 115(7): 928-935.",
];

refs.forEach((ref, i) => {
  children.push(referencePara(i + 1, ref));
});

children.push(emptyPara());

// ════════════════════════════════════════════════════════
// BUILD AND SAVE DOCUMENT
// ════════════════════════════════════════════════════════
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "default-numbering",
        levels: [
          {
            level: 0,
            format: LevelFormat ? LevelFormat.DECIMAL : "decimal",
            text: "%1.",
            alignment: AlignmentType.START,
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: {
            width: 12240,
            height: 15840,
          },
          margin: {
            top: 1440,
            right: 1440,
            bottom: 1440,
            left: 1440,
          },
        },
      },
      children: children,
    },
  ],
});

Packer.toBuffer(doc)
  .then((buffer) => {
    fs.writeFileSync(outputPath, buffer);
    console.log("Saved to:", outputPath);
  })
  .catch((err) => {
    console.error("Error generating document:", err);
    process.exit(1);
  });
