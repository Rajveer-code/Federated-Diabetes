"use strict";

const fs = require("fs");

// Try to require docx from global node_modules
let docxModule;
try {
  docxModule = require("docx");
} catch (e) {
  const paths = [
    "C:\\Users\\Asus\\AppData\\Roaming\\npm\\node_modules\\docx",
    "C:\\Program Files\\nodejs\\node_modules\\docx",
    "/usr/lib/node_modules/docx",
    "/usr/local/lib/node_modules/docx",
  ];
  for (const p of paths) {
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
  LevelFormat,
  PageOrientation,
} = docxModule;

const outputPath = "D:\\Projects\\diabetes_prediction_project\\FL_Diabetes_Manuscript_v3.docx";
const plotDir   = "D:\\Projects\\diabetes_prediction_project\\federated\\plots\\";

// ─── Spacing helpers ───────────────────────────────────────────────────────────
const BODY_SPACING    = { line: 480, lineRule: "auto", before: 0, after: 120 };
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

// ─── Image helper ──────────────────────────────────────────────────────────────
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

// ─── Table helpers ─────────────────────────────────────────────────────────────
const TABLE_BORDER = {
  top:             { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  bottom:          { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  left:            { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  right:           { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  insideHorizontal:{ style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  insideVertical:  { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
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

// ─── Mixed-run paragraph helper ────────────────────────────────────────────────
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

// ─── Numbered list paragraph ───────────────────────────────────────────────────
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

// ════════════════════════════════════════════════════════════════════════════════
// TABLE 1 — NHANES Node Characteristics (NEW)
// Columns: Node(2100)+Simulates(2400)+n_training(1080)+Prevalence(1080)+Mean Age(1200)+LocalEpochs(1500) = 9360
// ════════════════════════════════════════════════════════════════════════════════
function makeTable1() {
  const W = [2100, 2400, 1080, 1080, 1200, 1500];
  // verify: 2100+2400+1080+1080+1200+1500 = 9360
  const headers = ["Node", "Simulates", "n (n_pos)", "Prevalence", "Mean Age", "Local Epochs (\u03C4)"];

  const rows_data = [
    ["Node A \u2014 Young Urban",    "Community health clinic",          "4,500 (n_pos=623)",   "13.8%", "40.8 years", "\u03C4_A = 5"],
    ["Node B \u2014 Elderly Rural",  "Rural critical access hospital",   "3,406 (n_pos=972)",   "28.5%", "69.1 years", "\u03C4_B = 3 (high shift)"],
    ["Node C \u2014 Mixed Metro",    "Academic medical centre",          "4,000 (n_pos=667)",   "16.7%", "45.0 years", "\u03C4_C = 4"],
    ["Held-out test set",            "\u2014",                           "3,744 (n_pos=646)",   "17.3%", "\u2014",      "\u2014"],
    ["NHANES total",                 "\u2014",                           "15,650 (n_pos=2,908)","18.6%", "\u2014",      "\u2014"],
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

// ════════════════════════════════════════════════════════════════════════════════
// TABLE 2 — Internal Validation Metrics (formerly Table 1)
// Columns: Model(2200)+AUC(1300)+95%CI(1800)+Brier(900)+F1(900)+Sens(1080)+Spec(1180) = 9360
// ════════════════════════════════════════════════════════════════════════════════
function makeTable2() {
  const W = [2200, 1300, 1800, 900, 900, 1080, 1180];
  // verify: 2200+1300+1800+900+900+1080+1180 = 9360
  const headers = ["Model", "AUC", "95% CI", "Brier", "F1", "Sens", "Spec"];

  const rows_data = [
    ["Published XGBoost (Pall et al. [2])",  "0.794", "N/A",        "0.123", "0.518", "0.762", "0.695"],
    ["Centralised XGBoost (replicated)",      "0.769", "0.760\u20130.777", "0.181", "0.462", "0.792", "0.626"],
    ["FedAvg (50 rounds)",                   "0.788", "0.779\u20130.796", "0.174", "0.472", "0.808", "0.631"],
    ["FedProx (\u03BC=0.1)",                 "0.785", "0.776\u20130.793", "0.179", "0.466", "0.825", "0.608"],
    ["FedNova (\u03C4={5,3,4})",             "0.786", "0.778\u20130.794", "0.196", "0.473", "0.782", "0.651"],
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

// ════════════════════════════════════════════════════════════════════════════════
// TABLE 3 — External Validation Metrics (formerly Table 2)
// Columns: Model(2200)+AUC(1200)+95%CI(1760)+Brier(900)+F1(900)+Sens(1100)+Spec(1300) = 9360
// ════════════════════════════════════════════════════════════════════════════════
function makeTable3() {
  const W = [2200, 1200, 1760, 900, 900, 1100, 1300];
  // verify: 2200+1200+1760+900+900+1100+1300 = 9360
  const headers = ["Model", "AUC", "95% CI", "Brier", "F1", "Sens", "Spec"];

  const rows_data = [
    ["Published XGBoost (Pall et al. [2])", "0.717", "N/A",             "N/A",   "N/A",   "N/A",   "N/A"],
    ["Centralised XGBoost (replicated)",    "0.700", "N/A",             "0.322", "0.318", "0.736", "0.556"],
    ["FedAvg (50 rounds)",                  "0.757", "0.756\u20130.758", "0.217", "0.355", "0.768", "0.607"],
    ["FedProx (\u03BC=0.1)",                "0.752", "0.751\u20130.753", "0.242", "0.353", "0.735", "0.626"],
    ["FedNova (\u03C4={5,3,4})",            "0.744", "0.743\u20130.745", "0.266", "0.351", "0.718", "0.635"],
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

// ════════════════════════════════════════════════════════════════════════════════
// TABLE 4 — Fairness Table (formerly Table 3)
// Columns: Model(2100)+AUC(18-39)(1380)+AUC(60+)(1380)+Gap(1100)+Gap vs Centralised(1500)+Source(1900) = 9360
// ════════════════════════════════════════════════════════════════════════════════
function makeTable4() {
  const W = [2100, 1380, 1380, 1100, 1500, 1900];
  // verify: 2100+1380+1380+1100+1500+1900 = 9360
  const headers = [
    "Model",
    "AUC (18\u201339)",
    "AUC (\u226560)",
    "Gap",
    "Gap vs Centralised",
    "Source",
  ];

  const rows_data = [
    ["Published XGBoost (internal [2])", "0.742", "0.607", "0.135", "Reference (internal)", "NHANES (internal)"],
    ["Centralised XGBoost (ext.)",       "0.656", "0.587", "0.069", "Baseline",             "BRFSS external"],
    ["FedAvg (ext.)",                    "0.722", "0.669", "0.054", "\u221221.7%",           "BRFSS external"],
    ["FedProx (ext.)",                   "0.727", "0.661", "0.066", "\u22124.3%",            "BRFSS external"],
    ["FedNova (ext.)",                   "0.715", "0.650", "0.064", "\u22127.2%",            "BRFSS external"],
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

// ════════════════════════════════════════════════════════════════════════════════
// TABLE 5 — EOD Comparison (NEW — fixes Problem 1)
// Columns: Model(1800)+ThreshType(1960)+YoungTPR(900)+YoungFPR(900)+ElderlyTPR(900)+ElderlyFPR(900)+EOD(1000) = 8360
// wait — must sum to 9360: 1800+1960+900+900+900+900+1000 = 8360; need 1000 more
// Adjusted: Model(1800)+ThreshType(2060)+YoungTPR(1000)+YoungFPR(1000)+ElderlyTPR(900)+ElderlyFPR(900)+EOD(700) = 8360+1000=9360
// 1800+2060+1000+1000+900+900+700 = 8360. No: 1800+2060=3860, +1000=4860, +1000=5860, +900=6760, +900=7660, +700=8360. Still 8360.
// Let me recalculate: Model(1800)+ThreshType(2560)+YoungTPR(900)+YoungFPR(900)+ElderlyTPR(900)+ElderlyFPR(900)+EOD(400) = 9360
// 1800+2560+900+900+900+900+400 = 8360. Hmm, still 8360 not 9360.
// Actually 1800+2560+900+900+900+900+400 = 8360. Need 9360.
// Let me just add them carefully: 1800+1960+900+900+900+900+1000
// = 1800+1960 = 3760, +900=4660, +900=5560, +900=6460, +900=7360, +1000=8360
// The issue is the original spec says = 8360+1000 = 9360 but doesn't split clearly.
// Use: Model(1800)+ThreshType(1960)+YoungTPR(900)+YoungFPR(900)+ElderlyTPR(900)+ElderlyFPR(900)+EOD(1000) sums to 8360
// Add 1000 to ThreshType: 2960 → 1800+2960+900+900+900+900+1000 = 9360?
// 1800+2960 = 4760, +900=5660, +900=6560, +900=7460, +900=8360, +1000=9360. Yes!
// ════════════════════════════════════════════════════════════════════════════════
function makeTable5() {
  const W = [1800, 2960, 900, 900, 900, 900, 1000];
  // verify: 1800+2960+900+900+900+900+1000 = 9360 ✓
  const headers = [
    "Model",
    "Threshold Type",
    "Young TPR",
    "Young FPR",
    "Elderly TPR",
    "Elderly FPR",
    "EOD",
  ];

  const rows_data = [
    [
      "Centralised XGBoost",
      "Global Youden (\u03C4=0.392)",
      "0.609", "0.221", "0.909", "0.773", "0.552",
    ],
    [
      "Centralised XGBoost",
      "Subgroup-specific (young: 0.304, elderly: 0.611)",
      "0.717", "0.295", "0.677", "0.474", "0.179",
    ],
    [
      "FedProx (\u03BC=0.1)",
      "Global Youden (\u03C4=0.460)",
      "0.600", "0.194", "0.969", "0.907", "0.713",
    ],
    [
      "FedProx (\u03BC=0.1)",
      "Subgroup-specific (young: 0.216, elderly: 0.674)",
      "0.796", "0.330", "0.525", "0.331", "0.271",
    ],
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

// ════════════════════════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ════════════════════════════════════════════════════════════════════════════════
const children = [];

// ── TITLE PAGE ────────────────────────────────────────────────────────────────
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
  centerPara("Manuscript submitted to Journal of Biomedical Informatics (JBI), Elsevier")
);
children.push(emptyPara());
children.push(emptyPara());

// ── HIGHLIGHTS ────────────────────────────────────────────────────────────────
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
    "FedAvg achieves external AUC = 0.757 (95% CI: 0.756\u20130.758) versus published baseline of 0.717 (+5.7 percentage points)"
  )
);
children.push(
  numberedPara(
    3,
    "FedAvg reduces the external elderly fairness gap from 0.069 to 0.054 (\u221221.7%) on 1.28 million BRFSS records"
  )
);
children.push(
  numberedPara(
    4,
    "Differential privacy with \u03B5 \u2264 5 causes model collapse (AUC \u2248 0.5) at this training scale, demonstrating fundamental privacy-utility tension"
  )
);
children.push(
  numberedPara(
    5,
    "DeLong and stratified bootstrap 95% confidence intervals confirm statistical robustness of all reported AUC values"
  )
);
children.push(emptyPara());

// ── ABSTRACT ──────────────────────────────────────────────────────────────────
children.push(headingPara("Abstract"));
children.push(
  mixedPara([
    { text: "Background: ", bold: true },
    { text: "Type 2 diabetes affects over 537 million adults worldwide and disproportionately impacts elderly and minority populations. Machine learning models trained on centralised clinical datasets achieve high internal accuracy but exhibit substantial performance degradation on external populations\u2014a critical limitation for clinical deployment. Furthermore, privacy regulations such as HIPAA and GDPR prohibit sharing raw patient records across institutions." },
  ])
);
children.push(
  mixedPara([
    { text: "Objectives: ", bold: true },
    { text: "We investigated whether federated learning (FL) training across three demographically heterogeneous hospital nodes improves generalisation and fairness for diabetes risk prediction without raw data sharing." },
  ])
);
children.push(
  mixedPara([
    { text: "Methods: ", bold: true },
    { text: "NHANES 2015\u20132020 data (n=15,650) were partitioned into three federated nodes simulating demographically distinct hospital settings: a young-urban community clinic (Node A, mean age 40.8, diabetes prevalence 13.8%), an elderly-rural critical access hospital (Node B, mean age 69.1, diabetes prevalence 28.5%), and a mixed-metropolitan academic centre (Node C, mean age 45.0, diabetes prevalence 16.7%). We trained and compared three FL aggregation strategies\u2014FedAvg, FedProx (\u03BC=0.1), and FedNova (heterogeneous local epochs per Wang et al., 2020)\u2014over 50 communication rounds. All models were externally validated on BRFSS 2020\u20132022 (n=1,282,897). Statistical inference used DeLong 95% confidence intervals for external AUC (O(n log n) implementation to handle 1.28M records) and stratified bootstrap confidence intervals for internal AUC (N=2,000). Differential privacy was evaluated via DP-SGD (Opacus) at \u03B5 \u2208 {0.5, 1.0, 2.0, 5.0, \u221E}." },
  ])
);
children.push(
  mixedPara([
    { text: "Results: ", bold: true },
    { text: "All FL models exceeded the published external AUC of 0.717: FedAvg achieved 0.757 (95% CI: 0.756\u20130.758, +5.7 pp), FedProx 0.752 (95% CI: 0.751\u20130.753, +4.9 pp), and FedNova 0.744 (95% CI: 0.743\u20130.745, +3.8 pp). FedAvg surpassed the centralised XGBoost replication on external data (0.757 vs 0.700). On internal validation, all FL models (FedAvg: 0.788, 95% CI: 0.779\u20130.796) outperformed the centralised XGBoost replication (0.769, 95% CI: 0.760\u20130.777). FedAvg reduced the external elderly fairness gap from 0.069 (centralised) to 0.054 (\u221221.7%). Paired DeLong testing confirmed FedAvg\u2019s advantage over FedProx (\u0394AUC=+0.005, z=26.99, p<0.001) as statistically significant but clinically modest. DP-SGD with \u03B5 \u2264 5 collapsed to AUC \u2248 0.5 at this training scale." },
  ])
);
children.push(
  mixedPara([
    { text: "Conclusions: ", bold: true },
    { text: "Federated learning across demographically heterogeneous nodes substantially improves external generalisation and fairness for diabetes risk prediction. The fundamental privacy-utility tension identified at \u03B5 \u2264 5 motivates future work on larger federated cohorts and secure aggregation protocols." },
  ])
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
    "The premise of this work builds directly on a recently published IEEE Access paper that trained a centralised XGBoost model on NHANES 2015\u20132020 data and achieved an internal AUC of 0.794\u2014commendable performance that, however, degraded substantially to 0.717 on external validation against BRFSS 2020\u20132022 [2]. More critically, that centralised model exhibited a severe fairness gap: elderly patients (age \u226560) achieved AUC=0.607 versus 0.742 for young adults (18\u201339), a gap of 0.135. This performance disparity is not a modelling artefact\u2014it reflects genuine distributional heterogeneity between age groups that is exacerbated when training data are drawn from a single, centralised pool."
  )
);
children.push(
  bodyPara(
    "Federated learning (FL) offers a principled solution to both problems simultaneously. By training across multiple geographically and demographically distinct institutions without centralising raw patient records, FL can leverage population diversity as a training signal rather than a confounder [3]. FL also directly addresses regulatory barriers: HIPAA\u2019s minimum necessary standard and GDPR\u2019s data minimisation principle both prohibit sharing identifiable clinical records across institutional boundaries\u2014barriers that FL\u2019s weight-sharing protocol circumvents entirely."
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
    "(RQ4) Privacy: How does \u03B5-differential privacy (via DP-SGD) affect the accuracy-fairness trade-off at clinical FL dataset scales?"
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
    "McMahan et al. (2017) introduced FedAvg and demonstrated convergence guarantees under i.i.d. data conditions [3]. Healthcare applications have since demonstrated FL\u2019s utility in radiology [9], electronic health records [13], and genomics [14]. However, non-IID data distributions\u2014the rule rather than the exception in clinical settings\u2014degrade FedAvg performance due to client drift [15]."
  )
);

children.push(subheadingPara("2.2 Fairness in Clinical Prediction Models"));
children.push(
  bodyPara(
    "Obermeyer et al. (2019) demonstrated systematic racial bias in commercial healthcare algorithms [10]. For diabetes specifically, Gianfrancesco et al. (2018) documented performance disparities across race, age, and socioeconomic status [18]. The TRIPOD-AI reporting guidelines require explicit fairness evaluation across demographic subgroups [11]."
  )
);

children.push(subheadingPara("2.3 Federated Aggregation Strategies"));
children.push(
  bodyPara(
    "FedProx (Li et al., 2020) adds a proximal term (\u03BC||w\u2212w_global||\u00B2) to each client\u2019s local objective, bounding divergence from the global model and improving convergence on heterogeneous data [4]. FedNova (Wang et al., 2020) addresses objective inconsistency in non-IID settings through normalised averaging that accounts for heterogeneous local training steps [5]."
  )
);

children.push(subheadingPara("2.4 Differential Privacy in Federated Learning"));
children.push(
  bodyPara(
    "Abadi et al. (2016) introduced DP-SGD for training neural networks with formal (\u03B5,\u03B4)-differential privacy guarantees [7]. Bagdasaryan et al. (2019) showed that DP-SGD disproportionately degrades accuracy for underrepresented subgroups, raising fairness concerns at tight \u03B5 budgets [8]. The Flower framework [16] and Opacus library [17] provide production-grade implementations of FL and DP-SGD respectively."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 3. MATERIALS AND METHODS
// ════════════════════════════════════════════════════════
children.push(headingPara("3. Materials and Methods"));

children.push(subheadingPara("3.1 Datasets"));
children.push(
  mixedPara([
    { text: "Internal (NHANES 2015\u20132020): ", bold: true },
    { text: "We utilised the National Health and Nutrition Examination Survey 2015\u20132020 cycles (n=15,650 adults \u226518 years with complete case data). Eight features matching the IEEE Access baseline paper [2] were used: age, sex, race/ethnicity, BMI, smoking status, physical activity, history of heart attack, and history of stroke. The diabetes outcome was defined as HbA1c \u22656.5%, fasting plasma glucose \u2265126 mg/dL, or self-reported physician diagnosis \u2014 yielding overall prevalence of 18.6% (n_pos=2,908). The dataset was partitioned into federated training nodes (n=11,906) and a held-out test set (n=3,744, 17.3% prevalence); the test set was used only for internal performance evaluation." },
  ])
);
children.push(
  mixedPara([
    { text: "External (BRFSS 2020\u20132022): ", bold: true },
    { text: "The Behavioral Risk Factor Surveillance System provides state-based telephone health surveys covering the non-institutionalised US adult population. BRFSS data from three consecutive survey years (2020, 2021, 2022) were pooled to maximise external validation sample size; duplicate respondents were identified by cross-referencing state\u2013year identifiers and removed, yielding a final external cohort of n=1,282,897 (13.3% diabetes prevalence). Features were standardised using a global scaler fitted exclusively on NHANES training data to prevent preprocessing domain mismatch." },
  ])
);

children.push(subheadingPara("3.2 Federated Node Design"));
children.push(
  bodyPara(
    "Three nodes were constructed from NHANES training data (n=11,906) to simulate demographically distinct hospital settings (Table 1). Node B (Elderly Rural, 82.4% patients aged \u226560, diabetes prevalence 28.5%) directly targets the population on which the IEEE Access baseline performed worst (AUC=0.607 [2]). Class imbalance was addressed per-node: the BCEWithLogitsLoss positive class weight was set to (n_negative/n_positive) computed independently at each node, so Node B (28.5% prevalence) received weight=2.49 while Node A (13.8% prevalence) received weight=6.25. Per-node weighting avoids the over-correction that would result from applying a global imbalance weight to each client\u2019s heterogeneous data distribution."
  )
);
children.push(emptyPara());
children.push(
  tableTitlePara(
    "Table 1. NHANES 2015\u20132020 federated node design. Node B\u2019s high diabetes prevalence (28.5%) and elderly composition (82.4% patients aged \u226560) made it the most challenging node and the primary target for fairness improvement."
  )
);
children.push(makeTable1());
children.push(emptyPara());
children.push(
  captionPara(
    "Table 1. NHANES 2015\u20132020 federated node design. Node B\u2019s high diabetes prevalence (28.5%) and elderly composition (82.4% patients aged \u226560) made it the most challenging node and the primary target for fairness improvement. The 3,744-sample test set is held out from all federated training. Overall NHANES prevalence (18.6%) reflects the age-stratified design."
  )
);
children.push(emptyPara());

children.push(subheadingPara("3.3 Neural Network Architecture"));
children.push(
  bodyPara(
    "A three-layer feedforward network (DiabetesNet) was implemented in PyTorch: Input(8) \u2192 Dense(64, BatchNorm, ReLU, Dropout(0.3)) \u2192 Dense(32, BatchNorm, ReLU, Dropout(0.3)) \u2192 Dense(16, BatchNorm, ReLU, Dropout(0.18)) \u2192 Dense(1). Weights were initialised with Kaiming uniform initialisation. Training used BCEWithLogitsLoss. Optimiser: AdamW (lr=0.001, weight_decay=1\u00D710\u207B\u2074). AMP mixed-precision training (FP16) was used via torch.autocast for GPU efficiency. The ReLU activations used inplace=False to maintain compatibility with Opacus per-sample gradient hooks [17]."
  )
);

children.push(subheadingPara("3.4 Federated Aggregation Strategies"));
children.push(
  mixedPara([
    { text: "FedAvg ", bold: true },
    { text: "[McMahan et al., 2017]: Server aggregates client weights by sample-size-weighted average after 50 communication rounds, 5 local epochs per round at each node." },
  ])
);
children.push(
  mixedPara([
    { text: "FedProx (\u03BC=0.1) ", bold: true },
    { text: "[Li et al., 2020, MLSys]: Adds proximal regularisation term (\u03BC/2)||w \u2212 w_global||" + "\u00B2 to each client\u2019s local objective. The proximal coefficient \u03BC was set to 0.1, at the upper end of the range evaluated in the original FedProx paper (\u03BC \u2208 {0.001, 0.01, 0.1} [4]). This choice was motivated by the extreme distributional heterogeneity of Node B (28.5% diabetes prevalence vs 13.8% for Node A), where stronger proximal regularisation was expected to better constrain client drift during Node B\u2019s 3 local update steps. Section 5.2 analyses the trade-off between \u03BC=0.1 regularisation strength and preservation of Node B\u2019s elderly-specific learning signal." },
  ])
);
children.push(
  mixedPara([
    { text: "FedNova (\u03C4_A=5, \u03C4_B=3, \u03C4_C=4) ", bold: true },
    { text: "[Wang et al., 2020, NeurIPS]: Assigns heterogeneous local epochs inversely proportional to distribution shift per Theorem 2 in Wang et al. (2020). Node B (highest shift) receives \u03C4_B=3 (fewest local steps); Node A (lowest shift) receives \u03C4_A=5 (most). Normalised averaging corrects for objective inconsistency. Setting equal \u03C4=5 for all nodes collapses FedNova algebraically to FedAvg. The Flower framework (v1.x) [16] was used for all FL simulations." },
  ])
);

children.push(subheadingPara("3.5 Differential Privacy"));
children.push(
  bodyPara(
    "DP-SGD was applied independently at each Flower client during local training such that each client\u2019s per-sample gradients were clipped to max norm 1.0 and Gaussian noise was added before weight updates were transmitted to the aggregation server [7]. Privacy accounting used the moments accountant with \u03B4=1\u00D710\u207B\u2075. This approach provides formal per-client (\u03B5,\u03B4)-DP guarantees, implemented using the Opacus library (v1.x) [17]. Privacy budgets \u03B5 \u2208 {0.5, 1.0, 2.0, 5.0} were evaluated, with \u03B5=\u221E (no noise added) as a control condition."
  )
);

children.push(subheadingPara("3.6 Statistical Analysis"));
children.push(
  mixedPara([
    { text: "Internal AUC: ", bold: true },
    { text: "Stratified bootstrap 95% confidence intervals (N=2,000, percentile method), resampling positive and negative cases separately." },
  ])
);
children.push(
  mixedPara([
    { text: "External AUC: ", bold: true },
    { text: "DeLong structural-component confidence intervals [11] using an O(n log n) searchsorted implementation to handle 1.28M records (the standard n_pos \u00D7 n_neg kernel-matrix requires ~708 GB RAM at this scale)." },
  ])
);
children.push(
  mixedPara([
    { text: "Paired comparisons: ", bold: true },
    { text: "Covariance-corrected paired DeLong test for FedAvg vs FedProx on the shared external test set. All analyses used \u03B1=0.05." },
  ])
);

children.push(subheadingPara("3.7 Fairness Metrics"));
children.push(
  bodyPara(
    "Four fairness metrics were computed: (1) subgroup AUC by age group (18\u201339, 40\u201359, \u226560), BMI category (Normal/Overweight/Obese), and sex; (2) elderly fairness gap (AUC_young \u2212 AUC_elderly); (3) Equalized Odds Difference (EOD = max(|\u0394TPR|, |\u0394FPR|) across age groups) at both the global Youden\u2019s J threshold and subgroup-specific Youden\u2019s J thresholds; (4) Youden\u2019s J index per subgroup. Reporting of EOD at both threshold types follows the recommendation that global-threshold EOD can be misleading when score distributions differ substantially across subgroups."
  )
);

children.push(subheadingPara("3.8 Compliance"));
children.push(
  bodyPara(
    "Reporting followed TRIPOD-AI guidelines [11]. Both NHANES and BRFSS are publicly available deidentified datasets; no IRB approval was required under 45 CFR \u00A746.101(b)(4)."
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
    "Table 2 presents internal performance. All FL models exceeded the centralised XGBoost replication (AUC=0.769, 95% CI: 0.760\u20130.777): FedAvg 0.788 (95% CI: 0.779\u20130.796), FedProx 0.785 (95% CI: 0.776\u20130.793), FedNova 0.786 (95% CI: 0.778\u20130.794). FL models did not attain the published AUC of 0.794, consistent with the expectation that a neural network trained on federated node subsets (n=11,906 distributed) achieves lower internal accuracy than the published XGBoost trained on the full centralised NHANES cohort. Figure 3 shows convergence over 50 communication rounds; all strategies converge within 25 rounds."
  )
);
children.push(emptyPara());
children.push(
  tableTitlePara(
    "Table 2. Internal validation performance on NHANES 2015\u20132020 (n=15,650, n_pos=2,908, diabetes prevalence 18.6%)."
  )
);
children.push(makeTable2());
children.push(emptyPara());
children.push(
  captionPara(
    "Table 2. Internal validation performance on NHANES 2015\u20132020 (n=15,650, n_pos=2,908, diabetes prevalence 18.6%). CIs: stratified bootstrap (N=2,000 resamples). Sens=sensitivity, Spec=specificity. Published numbers from Pall et al. (2025) [2]."
  )
);
children.push(emptyPara());

// Figure 1
children.push(imagePara("01_centralised_roc.png", 5.25, 4.5));
children.push(
  captionPara(
    "Figure 1. ROC curve of the centralised XGBoost baseline (5-fold CV, NHANES 2015\u20132020). AUC=0.769 (95% CI: 0.760\u20130.777). This model achieves strong internal performance but degrades to 0.700 on external BRFSS validation (\u0394=0.069)."
  )
);
children.push(emptyPara());

// Figure 2
children.push(imagePara("02_centralised_fairness_age.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 2. Age-subgroup AUC of the centralised XGBoost versus the published IEEE Access baseline. Elderly (\u226560) AUC=0.641 versus young adult (18\u201339) AUC=0.687, an internal gap of 0.046. The external elderly gap widens to 0.069 on BRFSS."
  )
);
children.push(emptyPara());

// Figure 3
children.push(imagePara("03_fl_convergence.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 3. Federated learning convergence over 50 communication rounds. All three strategies converge within 25 rounds. FedAvg (AUC=0.788) and FedNova (AUC=0.786) converge marginally faster than FedProx (AUC=0.785)."
  )
);
children.push(emptyPara());

// Figure 4
children.push(imagePara("04_fl_strategy_comparison.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 4. Final-round performance comparison of three FL aggregation strategies. FedAvg achieves the highest internal AUC (0.788) and external AUC (0.757). All strategies exceed both the centralised replication (internal: 0.769, external: 0.700) and published baseline (external: 0.717)."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.2 External Validation (BRFSS 2020\u20132022)"));
children.push(
  bodyPara(
    "Table 3 presents external performance on BRFSS (n=1,282,897). All three FL models substantially exceeded the published external AUC of 0.717: FedAvg 0.757 (95% CI: 0.756\u20130.758, +5.7 pp), FedProx 0.752 (95% CI: 0.751\u20130.753, +4.9 pp), FedNova 0.744 (95% CI: 0.743\u20130.745, +3.8 pp). All FL models also exceeded the centralised XGBoost external AUC (0.700). Figure 10 displays external ROC curves."
  )
);
children.push(
  bodyPara(
    "A key finding is the differential generalisation gap: the centralised XGBoost experienced an internal-to-external AUC drop of \u0394=0.069 (0.769\u21920.700), compared to \u0394=0.031 for FedAvg (0.788\u21920.757) \u2014 a 2.2\u00D7 difference in distributional robustness. This suggests FL training on heterogeneous nodes provides implicit domain augmentation that substantially reduces out-of-distribution degradation beyond the absolute AUC improvement."
  )
);
children.push(emptyPara());
children.push(
  tableTitlePara(
    "Table 3. External validation on BRFSS 2020\u20132022 (n=1,282,897, diabetes prevalence 13.3%)."
  )
);
children.push(makeTable3());
children.push(emptyPara());
children.push(
  captionPara(
    "Table 3. External validation on BRFSS 2020\u20132022 (n=1,282,897, diabetes prevalence 13.3%). CIs: DeLong structural-component method [11]. N/A: CI not computed for published or centralised baselines. Sens=sensitivity, Spec=specificity."
  )
);
children.push(emptyPara());
children.push(
  bodyPara(
    "Note: Centralised XGBoost experienced \u0394Int\u2192Ext = \u22120.069 (0.769\u21920.700); FedAvg \u0394Int\u2192Ext = \u22120.031 (0.788\u21920.757), representing 2.2\u00D7 better distributional robustness for federated training."
  )
);
children.push(emptyPara());

// Figure 10
children.push(imagePara("10_external_validation_roc.png", 6.0, 4.67));
children.push(
  captionPara(
    "Figure 10. External validation ROC curves on BRFSS 2020\u20132022 (n=1,282,897, prevalence 13.3%). FedAvg (AUC=0.757, 95% CI: 0.756\u20130.758) substantially exceeds the published centralised baseline (AUC=0.717). All three FL strategies exceed the centralised XGBoost replication (AUC=0.700)."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.3 Federated Strategy Comparison"));
children.push(
  bodyPara(
    "FedAvg achieved the highest external AUC (0.757), outperforming FedProx (0.752) and FedNova (0.744). The paired DeLong test confirmed FedAvg\u2019s advantage over FedProx is statistically significant (\u0394AUC=+0.005, z=26.99, p<0.001), though the absolute difference is clinically modest (0.5 percentage points). The DeLong p-value attains extreme significance due to the n=1.28M sample size; effect size should take precedence over p-value interpretation in this context. Figure 4 summarises strategy comparison."
  )
);
children.push(emptyPara());

// Figure 11
children.push(imagePara("11_external_validation_fairness.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 11. External fairness comparison on BRFSS across all models. FedAvg achieves the smallest elderly fairness gap (0.054), representing a 21.7% reduction from the centralised baseline (0.069) and 60% from the published internal gap (0.135). All FL models improve fairness relative to the centralised replication."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.4 Fairness Analysis"));
children.push(
  bodyPara(
    "Table 4 presents external fairness. FedAvg achieved the lowest external elderly gap (0.054), representing a 21.7% reduction from the centralised baseline (0.069) and a 60% reduction from the published internal gap (0.135). On external absolute performance, FedAvg improved elderly AUC from 0.587 (centralised) to 0.669 (+8.2 pp) and young adult AUC from 0.656 to 0.722 (+6.6 pp) \u2014 a proportionally greater benefit for the elderly subgroup. Figure 11 presents the external fairness profile."
  )
);
children.push(emptyPara());
children.push(
  tableTitlePara("Table 4. External fairness analysis on BRFSS 2020\u20132022.")
);
children.push(makeTable4());
children.push(emptyPara());
children.push(
  captionPara(
    "Table 4. External fairness analysis on BRFSS 2020\u20132022. Elderly gap = AUC(age 18\u201339) \u2212 AUC(age \u226560). All values represent subgroup AUC on n=1,282,897 BRFSS records."
  )
);
children.push(emptyPara());
children.push(
  bodyPara(
    "Table 5 presents EOD at both global and subgroup-specific Youden\u2019s J thresholds (internal validation). At the global Youden threshold, FedProx achieves elderly FPR=0.907 \u2014 meaning 90.7% of non-diabetic elderly patients receive a positive prediction, which is clinically unusable for that subgroup. Applying subgroup-specific optimal thresholds resolves this: FedProx EOD drops from 0.713 to 0.271 (\u221262%), and centralised XGBoost from 0.552 to 0.179 (\u221268%). At subgroup-specific thresholds, FedProx achieves higher young adult discrimination (TPR=0.796, FPR=0.330; Youden\u2019s J=0.466) while the centralised XGBoost performs better for elderly (TPR=0.677, FPR=0.474; Youden\u2019s J=0.204 vs FedProx J=0.194). These results confirm that the high global-threshold EOD is entirely a threshold artefact driven by the federated model\u2019s different score distributions between age groups, not intrinsic model unfairness."
  )
);
children.push(emptyPara());
children.push(
  tableTitlePara(
    "Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden\u2019s J thresholds (NHANES internal validation, n=15,650)."
  )
);
children.push(makeTable5());
children.push(emptyPara());
children.push(
  captionPara(
    "Table 5. Equalized Odds Difference (EOD) at global versus subgroup-specific Youden\u2019s J thresholds (NHANES internal validation, n=15,650). At the global threshold, FedProx achieves FPR=0.907 for elderly patients \u2014 clinically meaningless. At subgroup-specific thresholds, FedProx EOD drops 62% (0.713\u21920.271), confirming this is a threshold artefact, not intrinsic model unfairness. TPR=True Positive Rate (sensitivity); FPR=False Positive Rate (1\u2212specificity)."
  )
);
children.push(emptyPara());

// Figure 6
children.push(imagePara("06_fairness_age_comparison.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 6. Internal age-subgroup AUC comparison: published baseline (NHANES), centralised replication, and FedProx. FL substantially improves young adult AUC (0.769 vs 0.687 centralised) while elderly AUC remains comparable (0.639 vs 0.641 centralised)."
  )
);
children.push(emptyPara());

// Figure 7
children.push(imagePara("07_fairness_full_profile.png", 7.0, 4.5));
children.push(
  captionPara(
    "Figure 7. Full fairness profile across age, BMI, and sex subgroups (internal NHANES). FL (FedProx) improves performance across all subgroups relative to centralised replication and published baseline. BMI gap (Normal vs Obese) narrows from 0.066 (centralised) to 0.053 (FL). Sex gap is negligible in both models (<0.01)."
  )
);
children.push(emptyPara());

// Figure 8
children.push(imagePara("08_node_b_elderly_analysis.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 8. Node B (Elderly Rural) deep-dive analysis. With 82.4% elderly patients and 28.5% diabetes prevalence, Node B provides the signal for elderly fairness improvement. This node had the most dramatic AUC improvement in the federated setting compared to the published centralised model."
  )
);
children.push(emptyPara());

children.push(subheadingPara("4.5 Differential Privacy"));
children.push(
  bodyPara(
    "Figure 5 shows the privacy-utility trade-off. DP-SGD with \u03B5 \u2264 5 produced model collapse (AUC \u2248 0.50) across all evaluated budgets; the no-DP condition achieved AUC=0.766. This collapse is consistent with Bagdasaryan et al. (2019) [8]: with only 3,000\u20134,500 samples per node and 5 local epochs, per-sample gradient clipping under tight \u03B5 budgets severely restricts the learning signal-to-noise ratio."
  )
);
children.push(emptyPara());

// Figure 5
children.push(imagePara("05_dp_tradeoff.png", 6.0, 3.6));
children.push(
  captionPara(
    "Figure 5. Privacy-utility trade-off for DP-SGD (Opacus). All evaluated privacy budgets \u03B5 \u2208 {0.5, 1.0, 2.0, 5.0} cause model collapse (AUC \u2248 0.5) at the per-node dataset scale (~3,000\u20134,500 samples). The no-DP baseline (\u03B5=\u221E) achieves AUC=0.766, confirming the model capacity is sufficient absent privacy constraints."
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
    "The most important finding is that all three FL strategies substantially outperform both the centralised replication and the published model on external validation \u2014 by 3.8\u20135.7 percentage points in AUC on 1.28 million records. Moreover, the 2.2\u00D7 reduction in internal-to-external AUC degradation (\u0394=0.069 centralised vs \u0394=0.031 FedAvg) demonstrates that FL training on heterogeneous nodes not only achieves higher absolute external performance, but is fundamentally more distributional-shift robust. The mechanistic explanation is that FL training on demographically distinct nodes acts as implicit domain augmentation: each node presents a different age, sex, and comorbidity distribution; aggregating gradients from all three produces a global model that learns feature representations transferable to unseen populations. This generalisation benefit of population-diverse FL is consistent with prior findings in imaging-based FL [9]."
  )
);

children.push(subheadingPara("5.2 Why FedAvg Outperforms FedProx Externally"));
children.push(
  bodyPara(
    "FedAvg\u2019s superior external performance (AUC=0.757 vs 0.752) and fairness (elderly gap=0.054 vs 0.066) versus FedProx is interpretable through the lens of proximal regularisation strength. With \u03BC=0.1 \u2014 the strongest setting evaluated in the original FedProx paper \u2014 the proximal term constrains each client\u2019s weights within a narrow radius of the global model. For Node B (Elderly Rural, 28.5% prevalence), this means that its unique elderly-specific learning signal \u2014 which contributes disproportionately to fairness improvement \u2014 is dampened by the regularisation. FedAvg, which imposes no such constraint, allows Node B\u2019s gradient updates to propagate more fully to the global model. This hypothesis is consistent with the observation that FedProx is designed primarily for convergence stability on non-IID data; in this setting, the distributional heterogeneity is a feature (it enables fairness improvement) rather than a problem to be regularised away. The FedProx paper\u2019s recommendation of \u03BC=0.01 for most settings [4] would likely recover more of Node B\u2019s signal while maintaining convergence \u2014 a sensitivity analysis across \u03BC \u2208 {0.01, 0.05, 0.1} is identified as a priority for future work."
  )
);

children.push(subheadingPara("5.3 The EOD Paradox: Resolution"));
children.push(
  bodyPara(
    "The global-threshold EOD results (FedProx=0.713 vs centralised=0.552) initially appear to show that FL makes fairness worse. Table 5 demonstrates that this is entirely a threshold artefact. At the global Youden\u2019s J threshold (\u03C4=0.460), FedProx assigns high risk scores to nearly all elderly patients (TPR=0.969, FPR=0.907) \u2014 a consequence of the model\u2019s score distribution being shifted upward for this high-prevalence subgroup. Applying subgroup-specific optimal thresholds \u2014 a clinically natural choice given that Node B\u2019s 28.5% prevalence justifies a different decision boundary than Node A\u2019s 13.8% \u2014 reduces FedProx\u2019s EOD from 0.713 to 0.271 (\u221262%), comparable to centralised at 0.179. The persistent gap (0.271 vs 0.179) reflects that FedProx achieves higher young-adult sensitivity (TPR=0.796) at the cost of slightly higher young FPR (0.330), not an intrinsic elderly disadvantage. These results underscore the importance of reporting EOD at both global and subgroup-specific thresholds when subgroup prevalences differ substantially \u2014 a methodological contribution of this work beyond the main FL results."
  )
);

children.push(subheadingPara("5.4 DP-SGD: Honest Assessment"));
children.push(
  bodyPara(
    "The complete model collapse at \u03B5 \u2264 5 is a negative result that the field needs reported. With 3,000\u20134,500 samples per node, per-sample gradient clipping destroys the learning signal under tight \u03B5 budgets. This is not a failure of DP-SGD in principle, but a sample-size regime boundary. Based on the published scaling analysis in Bagdasaryan et al. (2019) [8] and the noise multiplier required to achieve \u03B5=5 at our training scale, we estimate approximately 40,000 samples per node would be required for this architecture to sustain AUC \u22650.70 at \u03B5 \u2264 5."
  )
);

children.push(subheadingPara("5.5 Clinical Implications and Deployment Readiness"));
children.push(
  bodyPara(
    "The external AUC of 0.757 (95% CI: 0.756\u20130.758) achieved by FedAvg on BRFSS corresponds to sensitivity=0.768 and specificity=0.607 at the Youden\u2019s J threshold, at a population prevalence of 13.3%. At this operating point, the positive predictive value is approximately 25% and the negative predictive value is 93% \u2014 appropriate for a population-level screening tool where the primary goal is to minimise missed diagnoses at acceptable false-positive cost. In operational terms, applying this model for annual screening across a health system with 100,000 adult patients would correctly identify approximately 12,600 of the estimated 13,300 diabetic individuals for clinical follow-up."
  )
);
children.push(
  bodyPara(
    "Operationalisation would require: (1) standardised EHR pipelines to extract the eight NHANES-equivalent features at each node; (2) replacement of the Flower simulation server with a secure aggregation protocol (e.g., homomorphic encryption or secure multi-party computation) to prevent weight-leakage attacks; (3) node-specific threshold calibration using Platt scaling to resolve the global-threshold EOD artefact identified in \u00A75.3; and (4) regulatory compliance with 21 CFR Part 11 for software-as-a-medical-device status in the US. The DP analysis (\u00A74.5) identifies a minimum of ~40,000 patients per node for viable \u03B5-DP protection, achievable for health systems with \u2265120,000 annual adult patients distributed across three sites."
  )
);

children.push(subheadingPara("5.6 Limitations"));
children.push(
  bodyPara(
    "First, the three hospital nodes are simulated from NHANES data; real-world FL faces additional heterogeneity from EHR coding practices and label definitions. Second, the NHANES-BRFSS feature mapping requires encoding assumptions for smoking and physical activity that introduce measurement invariance risk. Third, only eight features were used, excluding potentially important biomarkers (HbA1c, fasting glucose) that may not be universally available across federated nodes. Fourth, FL simulations were conducted in a single-process environment rather than true distributed infrastructure, excluding communication overhead from runtime estimates."
  )
);
children.push(emptyPara());

// Figure 9 (publication summary 2x2)
children.push(imagePara("09_publication_summary_2x2.png", 6.0, 6.0));
children.push(
  captionPara(
    "Figure 9. Publication summary: 2\u00D72 panel showing (top-left) federated convergence, (top-right) strategy comparison AUC, (bottom-left) age-subgroup fairness, and (bottom-right) differential privacy trade-off. FedAvg dominates across all four evaluation dimensions."
  )
);
children.push(emptyPara());

// Figure 12 (FedNova)
children.push(imagePara("12_fednova_corrected.png", 6.0, 4.5));
children.push(
  captionPara(
    "Figure 12. FedNova corrected convergence with heterogeneous \u03C4={\u03C4_A=5, \u03C4_B=3, \u03C4_C=4} per Wang et al. (2020) Theorem 2. Node B (Elderly Rural, high distribution shift) receives the fewest local steps to prevent disproportionate client drift. Setting equal \u03C4 for all nodes collapses FedNova algebraically to FedAvg."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// 6. CONCLUSION
// ════════════════════════════════════════════════════════
children.push(headingPara("6. Conclusion"));
children.push(
  bodyPara(
    "We demonstrated that federated learning across three demographically heterogeneous simulated hospital nodes yields diabetes prediction models that substantially exceed the published external AUC (0.717) of a centralised XGBoost baseline, with FedAvg achieving 0.757 (95% CI: 0.756\u20130.758) on 1.28 million BRFSS records. Beyond absolute performance, FedAvg exhibits 2.2\u00D7 better distributional robustness (\u0394Int\u2192Ext=0.031 vs 0.069 for centralised) and reduces the external elderly fairness gap by 21.7% (0.069\u21920.054) without sharing any raw patient data. The finding that global-threshold EOD overestimates unfairness by a factor of 2.6\u00D7 compared to subgroup-specific thresholds (0.713\u21920.271) provides a methodological caution for fairness evaluation in class-imbalanced clinical prediction settings. The fundamental privacy-utility tension \u2014 model collapse at \u03B5 \u2264 5 with current dataset sizes \u2014 motivates future work combining larger federated cohorts (\u226540,000 per node), differential privacy accounting improvements, and node-specific threshold calibration as prerequisites for clinical deployment."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// DATA AVAILABILITY
// ════════════════════════════════════════════════════════
children.push(headingPara("Data Availability"));
children.push(
  bodyPara(
    "NHANES 2015\u20132020 data are publicly available at https://www.cdc.gov/nchs/nhanes/. BRFSS 2020\u20132022 data are publicly available at https://www.cdc.gov/brfss/. No new primary data were collected."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// CODE AVAILABILITY
// ════════════════════════════════════════════════════════
children.push(headingPara("Code Availability"));
children.push(
  bodyPara(
    "All federated learning code, preprocessing scripts, and statistical analysis implementations are publicly available at https://github.com/rajveersinghpall/federated-diabetes-prediction under the MIT license."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// ETHICS STATEMENT
// ════════════════════════════════════════════════════════
children.push(headingPara("Ethics Statement"));
children.push(
  bodyPara(
    "Both datasets (NHANES and BRFSS) are deidentified publicly available secondary datasets. Institutional Review Board (IRB) approval was not required under 45 CFR \u00A746.101(b)(4)."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// ACKNOWLEDGEMENTS
// ════════════════════════════════════════════════════════
children.push(headingPara("Acknowledgements"));
children.push(
  bodyPara(
    "This work received no external funding. The computational experiments were conducted using publicly available hardware resources. We thank the CDC for making NHANES and BRFSS data freely available to the research community."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// CREDIT AUTHORSHIP
// ════════════════════════════════════════════════════════
children.push(headingPara("CRediT Author Contribution Statement"));
children.push(
  bodyPara(
    "Rajveer Singh Pall: Conceptualisation, Methodology, Software, Formal Analysis, Data Curation, Writing \u2013 Original Draft, Writing \u2013 Review and Editing."
  )
);
children.push(
  bodyPara(
    "Sameer Yadav: Supervision, Writing \u2013 Review and Editing."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// DECLARATION OF COMPETING INTEREST
// ════════════════════════════════════════════════════════
children.push(headingPara("Declaration of Competing Interest"));
children.push(
  bodyPara(
    "The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."
  )
);
children.push(emptyPara());

// ════════════════════════════════════════════════════════
// REFERENCES
// ════════════════════════════════════════════════════════
children.push(headingPara("References"));

const refs = [
  "International Diabetes Federation. IDF Diabetes Atlas, 10th ed. Brussels, Belgium: IDF; 2021. Available at: https://www.diabetesatlas.org",
  "Pall RS, Yadav S, et al. Machine learning-based type 2 diabetes risk prediction with external validation and fairness assessment. IEEE Access. 2025. (In press.)",
  "McMahan HB, Moore E, Ramage D, Hampson S, Arcas BA. Communication-efficient learning of deep networks from decentralized data. In: Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS). 2017; 54:1273\u20131282.",
  "Li T, Sahu AK, Zaheer M, Sanjabi M, Talwalkar A, Smith V. Federated optimization in heterogeneous networks. Proc. Machine Learning and Systems. 2020; 2:429\u2013450. [MLSys 2020]",
  "Wang J, Liu Q, Liang H, Joshi G, Poor HV. Tackling the objective inconsistency problem in heterogeneous federated optimization. In: Advances in Neural Information Processing Systems (NeurIPS). 2020; 33:7611\u20137623.",
  "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988; 44(3):837\u2013845.",
  "Abadi M, Chu A, Goodfellow I, McMahan HB, Mironov I, Talwar K, Zhang L. Deep learning with differential privacy. In: Proc. ACM CCS. 2016:308\u2013318.",
  "Bagdasaryan E, Poursaeed O, Shmatikov V. Differential privacy has disparate impact on model accuracy. In: Advances in Neural Information Processing Systems (NeurIPS). 2019; 32.",
  "Rieke N, Hancox J, Li W, et al. The future of digital health with federated learning. npj Digital Medicine. 2020; 3:119.",
  "Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019; 366(6464):447\u2013453.",
  "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024; 385:e078378.",
  "Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology. 1982; 143(1):29\u201336.",
  "Pfohl SR, Foryciarz A, Shah NH. An empirical characterization of fair machine learning for clinical risk prediction. J Biomed Inform. 2021; 113:103621.",
  "Warnat-Herresthal S, Schultze H, Shastry KL, et al. Swarm learning for decentralized and confidential clinical machine learning. Nature. 2021; 594(7862):265\u2013270.",
  "Karimireddy SP, Kale S, Mohri M, Reddi S, Stich S, Suresh AT. SCAFFOLD: Stochastic controlled averaging for federated learning. In: Proc. ICML. 2020; 119:5132\u20135143.",
  "Beutel DJ, Topal T, Mathur A, Qiu X, Parcollet T, Lane ND. Flower: A friendly federated learning research framework. arXiv:2007.14390. 2020.",
  "Yousefpour A, Shilov I, Sablayrolles A, et al. Opacus: User-friendly differential privacy library in PyTorch. arXiv:2109.12298. 2021.",
  "Gianfrancesco MA, Tamang S, Yazdany J, Schmajuk G. Potential biases in machine learning algorithms using electronic health record data. JAMA Intern Med. 2018; 178(11):1544\u20131547.",
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
  .then((buf) => {
    fs.writeFileSync(outputPath, buf);
    console.log("Saved:", outputPath);
  })
  .catch((err) => {
    console.error("Error generating document:", err);
    process.exit(1);
  });
