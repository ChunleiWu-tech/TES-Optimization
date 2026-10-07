"""Build the V18 publication figures from the validated 0902 release.

Figures 1--4 and 6 retain the validated visual language of the preceding
release but load the corrected V18 service tables. Figure 5 is replaced by a
data-rich design-consequence analysis. Material-skeleton and nanoparticle
evidence remains in Supplementary Figure S8, where unlike experimental bases
are not ranked against each other.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
PP = HERE.parents[1]
ROOT = PP.parent
UP = ROOT / "upstream_package"
RES = UP / "results_v18"
RES19 = UP / "results_v19"
DATA = UP / "data"
OUT = Path(os.environ.get("TES_FIGURE_OUTPUT_ROOT", str(PP))).resolve()

sys.path.insert(0, str(HERE.parent))
import build_v16_figures as base  # noqa: E402


base.RES = RES
base.DATA = DATA
base.OUT = OUT
base.MAIN = OUT / "figures" / "main"
base.SUPP = OUT / "figures" / "supplementary"
base.CONTACT = OUT / "figures" / "contact_sheets"
base.TABLES = OUT / "tables"
for folder in (base.MAIN, base.SUPP, base.CONTACT, base.TABLES):
    folder.mkdir(parents=True, exist_ok=True)


SOURCE_MAP = {
    "codesign_results_v16": "codesign_results_v16.csv",
    "unit_pareto_front_v16": "unit_pareto_front_v16.csv",
    "fixed_service_results_v16": "fixed_service_results_v18.csv",
    "system_pareto_front_v16": "system_pareto_front_v18.csv",
    "system_pareto_exact_audit_v16": "system_pareto_exact_audit_v18.csv",
    "advantage_transfer_v16": "advantage_transfer_v18.csv",
    "pairwise_scale_order_v16": "pairwise_scale_order_v18.csv",
    "interaction_indices_v16": "interaction_indices_v16.csv",
    "topology_response_v16": "topology_response_v16.csv",
    "external_validation_trace_v16": "external_validation_trace_v16.csv",
    "external_validation_metrics_v16": "external_validation_metrics_v16.csv",
    "numerical_verification_v16": "numerical_verification_v16.csv",
    "particle_model_hierarchy_v16": "particle_model_hierarchy_v16.csv",
    "sampling_robustness_v16": "sampling_robustness_v16.csv",
    "bounded_robustness_v16": "bounded_robustness_v16.csv",
    "model_form_stress_v16": "model_form_stress_v16.csv",
    "standby_loss_v16": "standby_loss_v16.csv",
    "pareto_precision_sensitivity_v16": "pareto_precision_sensitivity_v18.csv",
    "screen_to_confirmation_v16": "screen_to_confirmation_v16.csv",
    "outlet_grade_threshold_sensitivity_v16": "outlet_grade_threshold_sensitivity_v18.csv",
    "module_scale_sensitivity_v16": "module_scale_sensitivity_v16.csv",
    "transport_temperature_sensitivity_v16": "transport_temperature_sensitivity_v16.csv",
    "service_monotonicity_audit_v16": "service_monotonicity_audit_v16.csv",
    "service_monotonicity_probe_v16": "service_monotonicity_probe_v18.csv",
    "decision_grid_refinement_v16": "decision_grid_refinement_v16.csv",
    "geometry_pool": "geometry_pool.csv",
    "evidence_coverage": "evidence_coverage.csv",
    "architecture_gate": "architecture_gate.csv",
    "nanoparticle_dose_response": "nanoparticle_dose_response.csv",
    "skeleton_within_source_effects": "skeleton_within_source_effects.csv",
}


def load_v18() -> dict[str, object]:
    tables: dict[str, object] = {
        key: pd.read_csv(RES / name) for key, name in SOURCE_MAP.items()
    }
    tables["pair_gate"] = pd.read_csv(DATA / "salt_skeleton_pair_registry.csv")
    tables["closure"] = json.loads(
        (RES / "reference_control_volume_closure_v16.json").read_text(encoding="utf-8")
    )
    tables["continuous_v18"] = pd.read_csv(RES / "01_continuous_service_closure.csv")
    tables["design_consequence_v18"] = pd.read_csv(
        RES / "02_pairwise_design_consequence.csv"
    )
    tables["design_summary_v18"] = pd.read_csv(
        RES / "03_design_consequence_summary.csv"
    )
    tables["integer_reclosure_v18"] = pd.read_csv(
        RES / "00_integer_reclosure_audit.csv"
    )
    tables["attenuation_decomposition_v19"] = pd.read_csv(
        RES19 / "scale_attenuation_decomposition_v19.csv"
    )
    tables["attenuation_summary_v19"] = pd.read_csv(
        RES19 / "scale_attenuation_summary_v19.csv"
    )
    tables["statistical_estimands_v19"] = pd.read_csv(
        RES19 / "statistical_estimands_v19.csv"
    )
    tables["integer_boundary_stability_v19"] = pd.read_csv(
        RES19 / "integer_boundary_stability_v19.csv"
    )
    tables["pareto_release_stability_v19"] = pd.read_csv(
        RES19 / "pareto_release_stability_v19.csv"
    )
    return tables


def copy_release_tables() -> None:
    """Copy a clean V18 analysis set and record inherited-table provenance."""
    used = sorted(set(SOURCE_MAP.values()))
    used += [
        "00_integer_reclosure_audit.csv",
        "00_service_correction_report.json",
        "00_service_dependent_rebuild_report.json",
        "00_monotonicity_probe_rebuild_report.json",
        "01_continuous_service_closure.csv",
        "02_pairwise_design_consequence.csv",
        "03_design_consequence_summary.csv",
        "04_dependency_audit_v18.json",
        "05_V18_QC_report.json",
        "06_upstream_release_audit_v18.json",
        "run_summary_v18.json",
    ]
    for name in sorted(set(used)):
        source = RES / name
        if source.is_file():
            shutil.copy2(source, base.TABLES / name)
    for source in (
        DATA / "salt_skeleton_pair_registry.csv",
        DATA / "external_validation_source_registry.csv",
    ):
        shutil.copy2(source, base.TABLES / source.name)
    for source in sorted(RES19.iterdir()):
        if source.is_file() and source.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(source, base.TABLES / source.name)
    provenance = []
    dependency = json.loads((RES / "04_dependency_audit_v18.json").read_text(encoding="utf-8"))
    retained = dependency["retained_v16_verification_tables"]
    for key, source_name in SOURCE_MAP.items():
        provenance.append(
            {
                "analysis_key": key,
                "release_file": source_name,
                "V18_recomputed": source_name.endswith("_v18.csv"),
                "retention_basis": retained.get(source_name, "V18 recomputation or unit-scale input"),
            }
        )
    pd.DataFrame(provenance).to_csv(
        base.TABLES / "V18_TABLE_PROVENANCE.csv", index=False
    )


def supplementary_10_statistical_design(t: dict[str, object]) -> dict[str, object]:
    pair = t["design_consequence_v18"]
    estimates = t["statistical_estimands_v19"]
    boundary = t["integer_boundary_stability_v19"]
    pareto = t["pareto_release_stability_v19"]
    fig, axs = base.five_panel_grid()

    ax = axs[0]
    base.panel(ax, 0)
    levels = ["Fluid", "Filled\nmodule", "Transient\nmodule", "Modular\nsystem"]
    columns = ["A_fluid_log", "A_filled_log", "A_dynamic_log", "A_system_log"]
    medians = np.array([pair[column].abs().median() for column in columns])
    x = np.arange(len(levels))
    ax.plot(x, medians, "o-", color="#3978A8", lw=1.7, ms=5)
    for xx, value in zip(x, medians):
        ax.text(xx, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, levels)
    ax.set_ylabel("Median absolute log contrast")
    ax.set_ylim(0, max(medians) * 1.22)
    base.clean(ax)

    ax = axs[1]
    base.panel(ax, 1)
    wanted = [
        "Median retained Solar-Salt-referenced contrast",
        "Median retained contrast across all material pairs",
    ]
    q = estimates.set_index("estimand").loc[wanted].reset_index()
    y = np.arange(len(q))
    point = q["point_estimate"].to_numpy(float)
    low = q["confidence_low"].to_numpy(float)
    high = q["confidence_high"].to_numpy(float)
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o", color="#3E9B65", capsize=3)
    ax.set_yticks(y, ["Solar-Salt\nreference", "All material\npairs"])
    ax.set_xlabel("Median retained contrast (%)")
    ax.set_xlim(44, 51)
    base.clean(ax)

    ax = axs[2]
    base.panel(ax, 2)
    service_names = [
        "Equal minimum-module-count frequency, 10 MW and 80 MWh",
        "Equal minimum-module-count frequency, 50 MW and 300 MWh",
    ]
    q = estimates.set_index("estimand").loc[service_names].reset_index()
    point = q["point_estimate"].to_numpy(float)
    low = q["confidence_low"].to_numpy(float)
    high = q["confidence_high"].to_numpy(float)
    y = np.arange(2)
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o", color="#E6862A", capsize=3)
    ax.set_yticks(y, ["10 MW, 80 MWh", "50 MW, 300 MWh"])
    ax.set_xlabel("Equal-count frequency (%)")
    ax.set_xlim(left=0)
    base.clean(ax)

    ax = axs[3]
    base.panel(ax, 3)
    q = boundary[boundary["scenario_id"].eq("ALL")].sort_values("perturbation_module_count")
    ax.semilogx(
        q["perturbation_module_count"],
        100.0 * q["classification_stable_fraction"],
        "o-",
        color="#D75452",
    )
    ax.axhline(100.0, color="#444444", lw=0.7, ls=":")
    ax.set_xlabel("Independent count perturbation, $\\varepsilon$ (modules)")
    ax.set_ylabel("Stable classifications (%)")
    ax.set_ylim(99.92, 100.005)
    base.clean(ax)

    ax = axs[4]
    base.panel(ax, 4)
    q = pareto[pareto["scenario_id"].eq("ALL")].copy()
    order = ["exact_nondominated_set", "objective_resolution_set"]
    q = q.set_index("front_definition").loc[order]
    values = 100.0 * q["jaccard_similarity"].to_numpy(float)
    bars = ax.bar(
        np.arange(2),
        values,
        color=["#8E6BB7", "#4FAF9F"],
        width=0.58,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(2), ["Exact\nnondominated set", "Objective-resolution\nset"])
    ax.set_ylabel("Pre-/post-reclosure membership Jaccard (%)")
    ax.set_ylim(0, 108)
    base.clean(ax)

    return base.finish(
        fig,
        base.SUPP / "Supplementary_Figure_S10_statistical_design_and_boundary_stability",
        5,
        [
            "02_pairwise_design_consequence.csv",
            "statistical_estimands_v19.csv",
            "integer_boundary_stability_v19.csv",
            "pareto_release_stability_v19.csv",
        ],
    )


def _equivalence_matrix(summary: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    q = summary[summary.summary_scope.eq("service_by_granularity")].copy()
    split = q.summary_label.str.split("::", expand=True)
    q["service"] = split[0]
    q["granularity"] = split[1]
    order = ["coarsest", "coarse", "intermediate", "fine", "finest"]
    pivot = q.pivot(index="service", columns="granularity", values="configuration_equivalent_fraction")
    pivot = pivot.reindex(index=list(base.SCENARIO_COLOR), columns=order)
    resolution_labels = []
    for label in order:
        value = 100.0 * float(
            q.loc[q.granularity.eq(label), "median_one_module_relative_resolution"].median()
        )
        resolution_labels.append(f"{value:.3f}%")
    return (
        100.0 * pivot.to_numpy(float),
        [base.SCENARIO_LABEL[x] for x in pivot.index],
        resolution_labels,
    )


def figure_5_v18(t: dict[str, object]) -> dict[str, object]:
    continuous = t["continuous_v18"].copy()
    pairs = t["design_consequence_v18"].copy()
    summary = t["design_summary_v18"].copy()
    audit = t["integer_reclosure_v18"].copy()
    fig, axs = base.composite_grid(6.9)

    # a: constraint competition at the continuous-count coordinate.
    ax = axs[0, 0]
    base.panel(ax, 0)
    sample = continuous.iloc[::2]
    energy = sample.governing_constraint.eq("energy_constrained")
    ax.scatter(
        sample.loc[energy, "N_P_cont"],
        sample.loc[energy, "N_E_cont"],
        s=7,
        color="#3978A8",
        alpha=0.20,
        edgecolors="none",
        label="Energy constrained",
        rasterized=True,
    )
    ax.scatter(
        sample.loc[~energy, "N_P_cont"],
        sample.loc[~energy, "N_E_cont"],
        s=14,
        color="#D75452",
        alpha=0.70,
        edgecolors="white",
        linewidth=0.25,
        label="Nominal power-rating constrained",
        rasterized=True,
    )
    lim = float(max(continuous.N_P_cont.max(), continuous.N_E_cont.max()))
    ax.plot([1, lim], [1, lim], "--", color="#333333", lw=0.8)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"Continuous nominal-power lower bound, $N_{P,\mathrm{cont}}$",
        ylabel=r"Continuous energy requirement, $N_{E,\mathrm{cont}}$",
    )
    base.clean(ax)
    base.legend_above(ax, 2)

    # b: nonzero fluid contrast can disappear only after the power floor governs.
    ax = axs[0, 1]
    base.panel(ax, 1)
    different = ~pairs.same_integer
    ax.scatter(
        100.0 * pairs.loc[different, "A_fluid_log"].abs(),
        pairs.loc[different, "delta_N_cont"],
        s=7,
        color="#BFC5CB",
        alpha=0.20,
        edgecolors="none",
        label="Different module counts",
        rasterized=True,
    )
    ax.scatter(
        100.0 * pairs.loc[~different, "A_fluid_log"].abs(),
        pairs.loc[~different, "delta_N_cont"],
        s=18,
        color="#D75452",
        alpha=0.72,
        edgecolors="white",
        linewidth=0.3,
        label="Same module count",
        zorder=4,
    )
    ax.set(
        yscale="symlog",
        xlabel=r"Absolute fluid-level log contrast, $|A_{fluid}|$ (%)",
        ylabel=r"Continuous module-count difference, $\Delta N_{cont}$",
    )
    base.clean(ax)
    base.legend_above(ax, 2)

    # c: the threshold margin separates equal and unequal module counts.
    ax = axs[0, 2]
    base.panel(ax, 2)
    for flag, colour, label in [
        (True, "#D75452", "Same module count"),
        (False, "#3978A8", "Different module counts"),
    ]:
        q = np.sort(pairs.loc[pairs.same_integer.eq(flag), "integer_boundary_margin"].to_numpy(float))
        y = 100.0 * np.arange(1, len(q) + 1) / len(q)
        ax.plot(q, y, color=colour, label=label)
    ax.axvline(0, color="#333333", lw=0.8, ls="--", label="Integer module-count threshold")
    ax.set(
        xscale="symlog",
        xlabel=r"Margin relative to integer module-count threshold, $M_{IB}$ (modules)",
        ylabel="Cumulative pairs (%)",
    )
    base.clean(ax)
    base.legend_above(ax, 2)

    # d: architecture granularity is reported as an observation, not tuned as a gate.
    ax = axs[1, 0]
    base.panel(ax, 3)
    values, rows, cols = _equivalence_matrix(summary)
    text_values = [
        ["—" if not np.isfinite(value) else f"{value:.2f}" for value in row]
        for row in values
    ]
    base.matrix(
        ax,
        values,
        rows,
        cols,
        "YlGnBu",
        0,
        max(9.0, float(np.nanmax(values))),
        fmt=".2f",
        text_values=text_values,
        cbar_label="Equal-module pairs (%)",
        text_fontsize=6.4,
    )
    ax.set(xlabel=r"Median one-module resolution, $100/N_{mid}$", ylabel="Power–energy requirement")

    # e: the two integer procedures differ only near the feasibility boundary.
    ax = axs[1, 1]
    base.panel(ax, 4)
    changed = audit[audit.count_delta_vs_frozen.ne(0)].copy()
    colours = changed.count_delta_vs_frozen.map({-1.0: "#3978A8", 1.0: "#D75452"})
    ax.scatter(
        1000.0 * changed.frozen_capacity_reproduction_error_pct,
        changed.adjacent_lower_energy_margin_pct,
        c=colours,
        s=25,
        alpha=0.72,
        edgecolors="white",
        linewidth=0.35,
    )
    ax.axvline(0, color="#444444", lw=0.7, ls=":")
    ax.axhline(0, color="#444444", lw=0.7, ls=":")
    handles = [
        Line2D([0], [0], marker="o", ls="", color="#3978A8", label="Count decreased by 1"),
        Line2D([0], [0], marker="o", ls="", color="#D75452", label="Count increased by 1"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=1,
        frameon=False,
        borderaxespad=0,
    )
    ax.set(
        xlabel=r"Installed-capacity difference ($\times 10^{-3}$ %)",
        ylabel="Adjacent-lower-count energy margin (%)",
    )
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    base.clean(ax)

    # f: compare exact nondominance with objective-resolution filtering.
    ax = axs[1, 2]
    base.panel(ax, 5)
    x = np.arange(2)
    width = 0.33
    before = np.array([267, 52])
    after = np.array([271, 52])
    ax.bar(x - width / 2, before, width, color="#9AA2AA", label="Monotonic feasibility search")
    ax.bar(x + width / 2, after, width, color="#3E9B65", label="Exhaustive integer enumeration")
    for xx, values in zip(x, zip(before, after)):
        for offset, value in zip((-width / 2, width / 2), values):
            ax.text(xx + offset, value + 4, str(value), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, ["Exact nondominated\nset", "Objective-resolution\nset"])
    ax.set_ylabel("Nondominated alternatives")
    ax.set_ylim(0, 310)
    base.clean(ax)
    base.legend_above(ax, 2)

    return base.finish(
        fig,
        base.MAIN / "Figure_05_design_consequence_boundary",
        6,
        [
            "01_continuous_service_closure",
            "02_pairwise_design_consequence",
            "03_design_consequence_summary",
            "00_integer_reclosure_audit",
            "system_pareto_exact_audit_v18",
            "system_pareto_front_v18",
        ],
    )


CAPTIONS = dict(base.CAPTIONS)
CAPTIONS["Figure 5"] = (
    "Constraint competition determines when a material contrast changes "
    "the minimum required module count. (a) Continuous energy requirement and nominal "
    "power-rating lower bound for 7,680 service cases. (b) Fluid-level contrast "
    "and continuous module-count difference for 15,360 matched material pairs. "
    "All 218 equal-count pairs are governed by the common nominal power-rating "
    "constraint. (c) Continuous requirement differences relative to the next integer module-count "
    "threshold reproduce all pairwise equal- and unequal-count classifications. (d) Equal-count frequency across service "
    "requirements and module granularities. (e) Monotonic feasibility search and exhaustive "
    "integer enumeration differ by one module in 67 cases; across all 7,680 cases, the maximum "
    "absolute installed-capacity difference is 0.00312%. (f) The two procedures yield exact "
    "nondominated sets of 267 and 271 alternatives, respectively, while retaining the same "
    "52 alternatives at the predefined objective-resolution thresholds."
)
CAPTIONS["Supplementary Figure S10"] = (
    "Statistical design and decision-boundary stability. (a) Median absolute log contrast "
    "across the four linked scales. Filling the module produces the dominant attenuation, "
    "finite-rate transport partially restores the contrast, and system integration attenuates "
    "it again. (b) Median retained contrast with 95% base-geometry cluster-bootstrap intervals "
    "for Solar-Salt-referenced and all-pair comparisons. (c) Equal minimum-module-count frequency "
    "with 95% cluster-bootstrap intervals for the two power–energy services. (d) Fraction of all "
    "15,360 pairwise classifications unchanged under independent worst-case perturbations of both "
    "continuous module counts. All classifications remain unchanged through 0.01 module; nine may "
    "change at 0.02 module. (e) Exact nondominated membership changes under integer reclosure, "
    "whereas all 52 alternatives retained at the predefined objective resolution are unchanged. "
    "Bootstrap intervals quantify sensitivity to the sampled base-geometry design space and do not "
    "represent material-property or experimental uncertainty."
)


def main() -> None:
    for folder in (base.MAIN, base.SUPP, base.CONTACT, base.TABLES):
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
    tables = load_v18()
    copy_release_tables()
    builders = (
        base.figure_1,
        base.figure_2,
        base.figure_3,
        base.figure_4,
        figure_5_v18,
        base.figure_6,
        base.supplementary_1,
        base.supplementary_2,
        base.supplementary_3,
        base.supplementary_4,
        base.supplementary_5,
        base.supplementary_6,
        base.supplementary_7,
        base.supplementary_8,
        base.supplementary_9,
        supplementary_10_statistical_design,
    )
    manifest = [builder(tables) for builder in builders]
    for record in manifest:
        resolved_sources = []
        for source in record["sources"]:
            if source in SOURCE_MAP:
                resolved_sources.append(SOURCE_MAP[source])
            elif (RES / f"{source}.csv").is_file() or (RES19 / f"{source}.csv").is_file():
                resolved_sources.append(f"{source}.csv")
            else:
                resolved_sources.append(source)
        record["sources"] = resolved_sources
    manifest_path = OUT / "figures" / "figure_manifest_v18.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "FIGURE_CAPTIONS_V18.md").write_text(
        "\n\n".join(f"{key}. {value}" for key, value in CAPTIONS.items()) + "\n",
        encoding="utf-8",
    )
    source_rows = []
    for record in manifest:
        for source in record["sources"]:
            source_rows.append(
                {
                    "figure_id": record["figure_id"],
                    "source_table_or_module": source,
                    "panel_count": record["panel_count"],
                }
            )
    pd.DataFrame(source_rows).to_csv(
        base.TABLES / "figure_source_map_v18.csv", index=False
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "main_figures": 6,
                "supplementary_figures": 10,
                "formats": ["pdf"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
