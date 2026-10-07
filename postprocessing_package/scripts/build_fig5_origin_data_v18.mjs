import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(".");
const results = path.join(root, "upstream_package", "results_v18");
const outputDir = path.join(root, "data_package");
const previewDir = path.join(root, "postprocessing_package", "tmp", "xlsx_previews");
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

async function readCSV(name) {
  const rows = parseCSV(await fs.readFile(path.join(results, name), "utf8"));
  const header = rows[0];
  return { header, rows: rows.slice(1).filter((row) => row.length && row[0] !== "") };
}

function typed(value) {
  if (value === "True" || value === "true") return true;
  if (value === "False" || value === "false") return false;
  if (value === "" || value === "nan" || value === "NaN") return null;
  const number = Number(value);
  if (Number.isFinite(number) && value.trim() !== "") return number;
  return value;
}

function select(table, columns, extra = null) {
  const indexes = columns.map((column) => {
    const index = table.header.indexOf(column);
    if (index < 0) throw new Error(`Missing column ${column}`);
    return index;
  });
  return table.rows.map((row) => {
    const values = indexes.map((index) => typed(row[index] ?? ""));
    return extra ? [...extra(row), ...values] : values;
  });
}

function columnName(number) {
  let n = number;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

const continuous = await readCSV("01_continuous_service_closure.csv");
const pairwise = await readCSV("02_pairwise_design_consequence.csv");
const summary = await readCSV("03_design_consequence_summary.csv");
const reclosure = await readCSV("00_integer_reclosure_audit.csv");
const exactOld = await readCSV("system_pareto_exact_audit_v16.csv");
const exactNew = await readCSV("system_pareto_exact_audit_v18.csv");
const decisionOld = await readCSV("system_pareto_front_v16.csv");
const decisionNew = await readCSV("system_pareto_front_v18.csv");

const sheets = [
  {
    name: "A_Constraints",
    title: "Figure 5a source data: competing continuous module requirements",
    source: "01_continuous_service_closure.csv",
    headers: [
      "case_id", "scenario_id", "fluid_name", "topology_id", "N_E_cont",
      "N_P_cont", "N_cont", "governing_constraint", "N_star", "root_relative_residual",
    ],
    rows: select(continuous, [
      "case_id", "scenario_id", "fluid_name", "topology_id", "N_E_cont",
      "N_P_cont", "N_cont", "governing_constraint", "N_star_old", "root_relative_residual",
    ]),
    formats: { E: "0.000000", F: "0.000000", G: "0.000000", I: "#,##0", J: "0.00E+00" },
  },
  {
    name: "BC_Pairwise",
    title: "Figure 5b-c source data: matched material contrast and integer-boundary outcome",
    source: "02_pairwise_design_consequence.csv",
    headers: [
      "pair_id", "scenario_id", "base_design_id", "topology_id", "fluid_name_i",
      "fluid_name_j", "A_fluid_log", "delta_N_cont", "next_integer_distance",
      "integer_boundary_margin", "Nstar_i", "Nstar_j", "same_integer",
      "governing_constraint_i", "governing_constraint_j", "boundary_identity_match",
      "one_module_relative_resolution", "pair_fate", "same_integer_flag",
      "boundary_identity_match_flag",
    ],
    rows: select(pairwise, [
      "pair_id", "scenario_id", "base_design_id", "topology_id", "fluid_name_i",
      "fluid_name_j", "A_fluid_log", "delta_N_cont", "next_integer_distance",
      "integer_boundary_margin", "Nstar_i", "Nstar_j", "same_integer",
      "governing_constraint_i", "governing_constraint_j", "boundary_identity_match",
      "one_module_relative_resolution", "pair_fate",
    ]).map((row) => [...row, row[12] ? 1 : 0, row[15] ? 1 : 0]),
    formats: { G: "0.000000", H: "0.000000", I: "0.000000", J: "0.000000", K: "#,##0", L: "#,##0", Q: "0.000000%", S: "0", T: "0" },
  },
  {
    name: "D_Resolution",
    title: "Figure 5d source data: service and one-module resolution",
    source: "03_design_consequence_summary.csv",
    headers: [
      "summary_scope", "summary_label", "n_pairs", "n_configuration_equivalent",
      "configuration_equivalent_fraction", "median_one_module_relative_resolution",
    ],
    rows: select(summary, [
      "summary_scope", "summary_label", "n_pairs", "n_configuration_equivalent",
      "configuration_equivalent_fraction", "median_one_module_relative_resolution",
    ]).filter((row) => row[0] === "service_by_granularity"),
    formats: { C: "#,##0", D: "#,##0", E: "0.000%", F: "0.000%" },
  },
  {
    name: "E_Reclosure",
    title: "Figure 5e source data: comparison of integer-sizing procedures",
    source: "00_integer_reclosure_audit.csv",
    headers: [
      "case_id", "scenario_id", "design_id", "fluid_model_id", "archived_count",
      "independent_count", "count_change", "archived_count_feasible",
      "capacity_reproduction_difference_pct", "adjacent_lower_energy_margin_pct",
      "corrected_capacity_margin_pct", "local_capacity_nondecreasing",
      "lower_search_truncated", "upper_search_truncated",
    ],
    rows: select(reclosure, [
      "case_id", "scenario_id", "design_id", "fluid_model_id",
      "frozen_parallel_unit_count", "current_minimum_count", "count_delta_vs_frozen",
      "frozen_count_feasible", "frozen_capacity_reproduction_error_pct",
      "adjacent_lower_energy_margin_pct", "corrected_capacity_margin_pct",
      "local_integer_capacity_nondecreasing", "lower_search_truncated", "upper_search_truncated",
    ]),
    formats: { E: "#,##0", F: "#,##0", G: "+0;-0;0", I: "0.000000", J: "0.000000", K: "0.000000" },
  },
];

const frontColumns = [
  "scenario_id", "design_id", "fluid_name", "topology_id", "parallel_unit_count",
  "installed_packed_volume_m3", "salt_inventory_t", "pump_energy_fraction",
  "capacity_oversize_pct",
];
const frontRows = [
  ...select(exactOld, frontColumns, () => ["exact_archived"]),
  ...select(exactNew, frontColumns, () => ["exact_independent"]),
  ...select(decisionOld, frontColumns, () => ["objective_resolution_archived"]),
  ...select(decisionNew, frontColumns, () => ["objective_resolution_independent"]),
];
sheets.push({
  name: "F_Pareto",
  title: "Figure 5f source data: exact and objective-resolution Pareto sets",
  source: "system_pareto_exact_audit_v16/v18.csv and system_pareto_front_v16/v18.csv",
  headers: ["evaluation_set", ...frontColumns],
  rows: frontRows,
  formats: { F: "#,##0", G: "#,##0.00", H: "#,##0.00", I: "0.000000", J: "0.000000" },
});

const workbook = Workbook.create();
const navy = "#17324D";
const teal = "#2A9D8F";
const pale = "#EAF2F5";

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1:B1").values = [["Item", "Definition"]];
readme.getRange("A2:B14").values = [
  ["Workbook", "Origin data for Figure 5: design-consequence boundary"],
  ["Dataset", "Validated numerical results archived on 2026-09-02"],
  ["Scientific question", "When does a matched thermophysical-property contrast change the required number of parallel packed-bed modules?"],
  ["Physical model", "A transient local thermal non-equilibrium packed-bed model is coupled to a continuous module coordinate and exact integer-boundary analysis."],
  ["Service constraints", "Nominal module power rating, required energy Pτ, and normalized outlet-temperature criterion g_out ≥ 0.90."],
  ["N_E", "Self-consistent continuous module requirement imposed by deliverable energy."],
  ["N_P", "Continuous lower bound imposed by the nominal module power rating."],
  ["N_cont", "max(N_E, N_P, 1); an analysis coordinate, not a fractional physical installation."],
  ["M_IB", "ΔN_cont minus the distance from the smaller continuous count to the next integer boundary."],
  ["Key result", "218 matched pairs have the same module count; all are governed by the common nominal power-rating constraint."],
  ["Integer-procedure comparison", "Monotonic feasibility search and exhaustive integer enumeration differ by one module in 67 of 7,680 cases; the maximum absolute installed-capacity difference is 0.00312%."],
  ["Nondominated alternatives", "The two integer procedures yield exact nondominated sets of 267 and 271 alternatives, respectively, while retaining the same 52 alternatives at predefined objective-resolution thresholds."],
  ["Evidence requirement", "System-level comparison of alternative skeletons requires matched thermal, hydraulic, chemical, wetting, and cycling measurements."],
];
readme.getRange("A1:B1").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
readme.getRange("A2:A14").format = { fill: pale, font: { bold: true, color: navy } };
readme.getRange("A1:B14").format.wrapText = true;
readme.getRange("A1:B14").format.verticalAlignment = "top";
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 95;
readme.getRange("1:1").format.rowHeight = 24;
readme.freezePanes.freezeRows(1);
readme.tables.add("A1:B14", true, "ReadmeTable").style = "TableStyleMedium2";

for (const spec of sheets) {
  const sheet = workbook.worksheets.add(spec.name);
  sheet.showGridLines = false;
  const matrix = [spec.headers, ...spec.rows];
  const lastColumn = columnName(spec.headers.length);
  sheet.getRange(`A1:${lastColumn}${matrix.length}`).values = matrix;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}${matrix.length}`).format.font = { name: "Arial", size: 9 };
  sheet.getRange(`A1:${lastColumn}${matrix.length}`).format.verticalAlignment = "top";
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: navy,
    font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}${matrix.length}`).format.autofitColumns();
  sheet.getRange(`A1:${lastColumn}${Math.min(matrix.length, 40)}`).format.autofitRows();
  sheet.getRange("1:1").format.rowHeight = 38;
  for (let column = 1; column <= spec.headers.length; column += 1) {
    const letter = columnName(column);
    const maxWidth = ["case_id", "pair_id", "design_id"].includes(spec.headers[column - 1]) ? 34 : 24;
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = Math.min(maxWidth, Math.max(10, maxWidth));
  }
  for (const [letter, format] of Object.entries(spec.formats)) {
    sheet.getRange(`${letter}2:${letter}${matrix.length}`).format.numberFormat = format;
  }
  if (spec.name === "D_Resolution") {
    sheet.getRange("E:F").format.columnWidth = 36;
  }
  if (spec.name === "E_Reclosure") {
    sheet.getRange("I:N").format.columnWidth = 30;
  }
  sheet.freezePanes.freezeRows(1);
  sheet.tables.add(`A1:${lastColumn}${matrix.length}`, true, `${spec.name.replace(/_/g, "")}Table`).style = "TableStyleMedium2";
}

const qa = workbook.worksheets.add("QA");
qa.showGridLines = false;
qa.getRange("A1:D1").values = [["Check", "Observed", "Expected", "Status"]];
qa.getRange("A2:A8").values = [
  ["Continuous service cases"],
  ["Matched material pairs"],
  ["Equal-module pairs"],
  ["Boundary identity matches"],
  ["Integer counts changed"],
  ["Exact nondominated alternatives from exhaustive enumeration"],
  ["Objective-resolution alternatives from exhaustive enumeration"],
];
qa.getRange("B2:B8").formulas = [
  ["=COUNTA('A_Constraints'!A2:A7681)"],
  ["=COUNTA('BC_Pairwise'!A2:A15361)"],
  ["=SUM('BC_Pairwise'!S2:S15361)"],
  ["=SUM('BC_Pairwise'!T2:T15361)"],
  ["=COUNTIF('E_Reclosure'!G2:G7681,\"<>0\")"],
  ["=COUNTIF('F_Pareto'!A2:A643,\"exact_independent\")"],
  ["=COUNTIF('F_Pareto'!A2:A643,\"objective_resolution_independent\")"],
];
qa.getRange("C2:C8").values = [[7680], [15360], [218], [15360], [67], [271], [52]];
qa.getRange("D2").formulas = [["=IF(B2=C2,\"PASS\",\"FAIL\")"]];
qa.getRange("D2:D8").fillDown();
qa.getRange("A1:D1").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
qa.getRange("A2:A8").format = { fill: pale, font: { bold: true, color: navy } };
qa.getRange("D2:D8").conditionalFormats.add("containsText", { text: "PASS", format: { fill: "#D9EAD3", font: { color: "#276749", bold: true } } });
qa.getRange("D2:D8").conditionalFormats.add("containsText", { text: "FAIL", format: { fill: "#F4CCCC", font: { color: "#9C1C1C", bold: true } } });
qa.getRange("A1:D8").format.font = { name: "Arial", size: 10 };
qa.getRange("A:A").format.columnWidth = 52;
qa.getRange("B:D").format.columnWidth = 20;
qa.freezePanes.freezeRows(1);
qa.tables.add("A1:D8", true, "QATable").style = "TableStyleMedium2";

const previews = [
  ["README", "A1:B14"],
  ["A_Constraints", "A1:J18"],
  ["BC_Pairwise", "A1:T16"],
  ["D_Resolution", "A1:F12"],
  ["E_Reclosure", "A1:N16"],
  ["F_Pareto", "A1:J16"],
  ["QA", "A1:D8"],
];
for (const [sheetName, range] of previews) {
  const blob = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await blob.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "table",
  range: "QA!A1:D8",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 6,
});
console.log(inspection.ndjson);
if (inspection.ndjson.includes('"FAIL"')) {
  throw new Error("Workbook QA contains a failed check");
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "04_Fig5_origin_data.xlsx");
await xlsx.save(outputPath);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
console.log(JSON.stringify({ status: "PASS", output: outputPath, sheets: ["README", ...sheets.map((sheet) => sheet.name), "QA"] }, null, 2));
