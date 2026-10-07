"""Build the V16 publication figures from the frozen upstream result tables.

The script deliberately separates scientific roles across figures.  No panel
has a title.  Panel letters use one registered location, legends are placed
above their axes. Every figure is written directly from the same Matplotlib
object as a single vector PDF. Temporary visual-QA renders are never stored in
the submission package.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PP = HERE.parents[1]
ROOT = PP.parent
OUT = Path(os.environ.get("TES_FIGURE_OUTPUT_ROOT", str(PP))).resolve()
# The 08 submission package keeps one self-contained, locked upstream package.
# Never fall back to a neighbouring historical run: every figure must be
# reproducible from the data bundled under this package root.
UP = ROOT / "upstream_package"
RES = UP / "results_v16"
DATA = UP / "data"
MAIN = OUT / "figures" / "main"
SUPP = OUT / "figures" / "supplementary"
CONTACT = OUT / "figures" / "contact_sheets"
TABLES = OUT / "tables"
for path in (MAIN, SUPP, CONTACT, TABLES):
    path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(UP))
from tespub.fluid_properties import fluid_properties, get_model_spec  # noqa: E402

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 8.0,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.35,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

FLUID_ORDER = [
    "SOLAR_SALT_60_40_V10",
    "NREL_MGCL2_KCL_NACL_BASELINE_V10",
    "AIP_FLINAK_2017_V10",
    "AIP_LINAK_CARBONATE_2017_V10",
    "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10",
]
FLUID_SHORT = {
    "SOLAR_SALT_60_40_V10": "Solar Salt",
    "NREL_MGCL2_KCL_NACL_BASELINE_V10": "MgCl$_2$–KCl–NaCl",
    "AIP_FLINAK_2017_V10": "FLiNaK",
    "AIP_LINAK_CARBONATE_2017_V10": "Li–Na–K carbonate",
    "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10": "Reciprocal salt",
}
FLUID_ANNOTATION_SHORT = {
    "SOLAR_SALT_60_40_V10": "Solar Salt",
    "NREL_MGCL2_KCL_NACL_BASELINE_V10": "MgCl$_2$ mix",
    "AIP_FLINAK_2017_V10": "FLiNaK",
    "AIP_LINAK_CARBONATE_2017_V10": "Li–Na–K carbonate",
    "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10": "Reciprocal salt",
}
PALETTE = {
    "SOLAR_SALT_60_40_V10": "#3978A8",
    "NREL_MGCL2_KCL_NACL_BASELINE_V10": "#E6862A",
    "AIP_FLINAK_2017_V10": "#3E9B65",
    "AIP_LINAK_CARBONATE_2017_V10": "#D75452",
    "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10": "#8E6BB7",
}
TOPO_COLOR = {"none": "#3978A8", "fine_hot": "#3E9B65", "coarse_hot": "#D75452"}
SCENARIO_COLOR = {"S10MW_8H": "#3978A8", "S50MW_6H": "#E6862A"}
SCENARIO_LABEL = {"S10MW_8H": "10 MW, 80 MWh", "S50MW_6H": "50 MW, 300 MWh"}
CANDIDATE_SHORT = {
    "CERAMIC_SPHERE_PACKED_BED": "Ceramic bed",
    "SOLAR_SALT_NEAT_PCM": "Neat Solar Salt",
    "CEG_SOLAR_SALT_60CR": "Expanded graphite",
    "SIC_FOAM_SOLAR_SALT": "SiC foam",
    "NI_FOAM_SOLAR_SALT_G1": "Ni foam",
    "CU_FOAM_SOLAR_SALT_G1": "Cu foam",
}
ARCHITECTURE_SHORT = {
    "sensible_packed_bed_LTNE": "Sensible packed bed",
    "encapsulated_pcm_packed_bed": "Encapsulated phase-change material",
    "finned_pcm": "Finned phase-change material",
    "open_cell_foam_pcm": "Open-cell phase-change material foam",
    "structured_channel": "Structured channel",
    "cascaded_units": "Cascaded units",
}
NANOPARTICLE_SHORT = {
    "SiO2": "SiO$_2$",
    "SiO2-Al2O3": "SiO$_2$–Al$_2$O$_3$",
    "CuO": "CuO",
    "TiO2": "TiO$_2$",
    "graphene nanoplatelets": "Graphene nanoplatelets\n(GNP)",
}
LETTERS = list("abcdefghi")


def panel(ax: plt.Axes, index: int) -> None:
    ax.text(-0.135, 1.055, LETTERS[index], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.5, fontweight="bold", clip_on=False)


def clean(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color="#D5D9DE", alpha=0.62, linewidth=0.55)
        ax.set_axisbelow(True)


def legend_above(ax: plt.Axes, ncol: int = 2, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = max(1, min(ncol, 3, len(handles)))
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.015),
              ncol=ncol, frameon=False, columnspacing=0.65, handletextpad=0.35,
              borderaxespad=0.0, **kwargs)


def composite_grid(height: float = 6.8, ncols: int = 3) -> tuple[plt.Figure, np.ndarray]:
    """Create a compact 2-row publication grid with stable panel widths."""
    fig, axs = plt.subplots(2, ncols, figsize=(11.0, height), layout="constrained")
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=0.018, h_pad=0.028, wspace=0.035, hspace=0.075)
    return fig, axs


def five_panel_grid(height: float = 6.8) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a 3-over-2 composite; the final validation panel spans two columns."""
    fig = plt.figure(figsize=(11.0, height), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    axs = [
        fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1:]),
    ]
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=0.018, h_pad=0.028, wspace=0.035, hspace=0.075)
    return fig, axs


def figure_legend(fig: plt.Figure, handles: list[Line2D], labels: list[str], ncol: int = 3) -> None:
    """Place a shared legend in a dedicated header, never over a data panel."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.99),
               ncol=ncol, frameon=False, columnspacing=0.8, handletextpad=0.35,
               borderaxespad=0.0, fontsize=6.8)


def candidate_short(value: object) -> str:
    key = str(value)
    return CANDIDATE_SHORT.get(key, key.replace("_", " "))


def architecture_short(value: object) -> str:
    key = str(value)
    return ARCHITECTURE_SHORT.get(key, key.replace("_", " "))


def nanoparticle_short(value: object) -> str:
    key = str(value)
    return NANOPARTICLE_SHORT.get(key, key)


def finish(fig: plt.Figure, stem: Path, panels: int, sources: list[str]) -> dict[str, object]:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "figure_id": stem.name,
        "panel_count": panels,
        "pdf": str(stem.with_suffix(".pdf").relative_to(OUT)),
        "formats": ["pdf"],
        "font": "Arial",
        "panel_titles": 0,
        "panel_letter_position_axes": [-0.135, 1.055],
        "sources": sources,
    }


def matrix(ax: plt.Axes, values: np.ndarray, rows: list[str], cols: list[str],
           cmap, vmin: float, vmax: float, fmt: str = ".2f", text_values=None,
           cbar_label: str | None = None, center: float | None = None,
           text_fontsize: float = 6.6) -> None:
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else None
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=None if norm else vmin,
                   vmax=None if norm else vmax, norm=norm)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            label = text_values[i][j] if text_values is not None else format(values[i, j], fmt)
            rgba = im.cmap(im.norm(values[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            colour = "#FFFFFF" if luminance < 0.48 else "#1A1A1A"
            ax.text(j, i, label, ha="center", va="center", fontsize=text_fontsize, color=colour)
    ax.set_xticks(range(len(cols)), cols, rotation=28, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cb.set_label(cbar_label)


def load() -> dict[str, object]:
    names = [
        "codesign_results_v16.csv", "unit_pareto_front_v16.csv",
        "fixed_service_results_v16.csv", "system_pareto_front_v16.csv",
        "system_pareto_exact_audit_v16.csv", "advantage_transfer_v16.csv",
        "pairwise_scale_order_v16.csv",
        "interaction_indices_v16.csv", "topology_response_v16.csv",
        "external_validation_trace_v16.csv", "external_validation_metrics_v16.csv",
        "numerical_verification_v16.csv", "particle_model_hierarchy_v16.csv",
        "sampling_robustness_v16.csv", "bounded_robustness_v16.csv",
        "model_form_stress_v16.csv", "standby_loss_v16.csv",
        "pareto_precision_sensitivity_v16.csv", "screen_to_confirmation_v16.csv",
        "outlet_grade_threshold_sensitivity_v16.csv", "module_scale_sensitivity_v16.csv",
        "transport_temperature_sensitivity_v16.csv", "service_monotonicity_audit_v16.csv",
        "service_monotonicity_probe_v16.csv", "decision_grid_refinement_v16.csv",
        "geometry_pool.csv", "evidence_coverage.csv", "architecture_gate.csv",
        "nanoparticle_dose_response.csv", "skeleton_within_source_effects.csv",
    ]
    out: dict[str, object] = {Path(n).stem: pd.read_csv(RES / n) for n in names}
    out["pair_gate"] = pd.read_csv(DATA / "salt_skeleton_pair_registry.csv")
    out["closure"] = json.loads((RES / "reference_control_volume_closure_v16.json").read_text(encoding="utf-8"))
    return out


def copy_release_tables() -> None:
    for p in sorted(RES.iterdir()):
        if p.is_file() and p.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(p, TABLES / p.name)
    for p in (DATA / "salt_skeleton_pair_registry.csv", DATA / "external_validation_source_registry.csv"):
        shutil.copy2(p, TABLES / p.name)


def figure_1(t: dict[str, object]) -> dict[str, object]:
    tr = t["external_validation_trace_v16"]
    met = t["external_validation_metrics_v16"]
    nv = t["numerical_verification_v16"]
    ph = t["particle_model_hierarchy_v16"]
    conf = t["screen_to_confirmation_v16"]
    closure = t["closure"]
    fig, axs = composite_grid()

    ax = axs[0, 0]; panel(ax, 0)
    ax.fill_between(tr.time_min, tr.observed_radial_min_K, tr.observed_radial_max_K,
                    color="#BCD0E6", alpha=0.65, label="Observed radial envelope")
    ax.plot(tr.time_min, tr.observed_area_weighted_outlet_K, color="#285A8E", label="Observed outlet")
    ax.plot(tr.time_min, tr.predicted_1D_outlet_K, color="#E46C2A", label="One-dimensional two-temperature model")
    phase = tr.loc[tr.phase.str.contains("cooling"), "time_min"].min()
    ax.axvline(phase, color="#333333", ls=":", lw=0.8)
    ax.set(xlabel="Time (min)", ylabel="Outlet temperature (K)")
    clean(ax); legend_above(ax, 2)

    ax = axs[0, 1]; panel(ax, 1)
    order = ["calibration", "validation", "diagnostic"]
    p = met.set_index("role").loc[order]
    x = np.arange(3)
    ax.bar(x, p.RMSE_K, color=["#3E9B65", "#D75452", "#8C929A"], width=0.62)
    ax.set_xticks(x, ["Heating\ncalibration", "Independent\ncooling test", "Full thermal\ncycle"])
    ax.set_ylabel("Root-mean-square error (K)")
    ax2 = ax.twinx()
    ax2.plot(x, p.radial_envelope_coverage_pct, "o-", color="#285A8E", label="Envelope coverage")
    ax2.set_ylabel("Radial-envelope coverage (%)")
    ax2.set_ylim(70, 102)
    ax2.spines["top"].set_visible(False)
    clean(ax)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), frameon=False)

    ax = axs[0, 2]; panel(ax, 2)
    grid = nv[nv.verification_case == "grid_energy_convergence"]
    for topo, marker, colour in [("homogeneous", "o", "#3978A8"), ("graded_two_layer", "s", "#3E9B65")]:
        q = grid[grid.topology == topo].sort_values("n_cells")
        finest_high_grade = float(q.loc[q.n_cells.idxmax(), "high_grade_discharge_fraction"])
        high_grade_relative_error_pct = 100.0 * (
            q.high_grade_discharge_fraction / finest_high_grade - 1.0
        )
        ax.plot(q.n_cells, q.t90_relative_error_vs_finest_pct.abs(), marker=marker,
                color=colour, label=f"$t_{{90}}$, {'homogeneous' if topo == 'homogeneous' else 'graded'}")
        ax.plot(q.n_cells, high_grade_relative_error_pct.abs(), marker=marker,
                color=colour, ls="--", label=f"Deliverable fraction, {'homogeneous' if topo == 'homogeneous' else 'graded'}")
    ax.set(xscale="log", yscale="log", xlabel="Axial cells", ylabel="Absolute relative error (%)")
    ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096], [64, 128, 256, 512, 1024, 2048, 4096])
    ax.set_xlim(54, 4800)
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 0]; panel(ax, 3)
    ax.scatter(ph.Bi_radius, ph.lumped_error_vs_radial_pct, marker="x", s=40,
               color="#3978A8", label="Lumped capacitance")
    ax.scatter(ph.Bi_radius, ph.effective_resistance_error_vs_radial_pct, marker="o", s=30,
               color="#E6862A", label="Effective resistance")
    ax.axhspan(-3, 3, color="#3E9B65", alpha=0.12, label="±3% acceptance band")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set(xlabel="Particle Biot number, $Bi_r$", ylabel="$t_{90}$ error relative to radial finite-volume model (%)")
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 1]; panel(ax, 4)
    x = np.arange(2)
    reported = [closure["reported_filler_MWh"], closure["reported_salt_inventory_MWh"]]
    resolved = [closure["derived_filler_MWh"], closure["derived_geometric_pore_salt_MWh"]]
    unresolved = [max(0, reported[0]-resolved[0]), closure["unresolved_reported_salt_inventory_MWh"]]
    ax.bar(x, resolved, color="#3978A8", label="Geometry-derived inventory")
    ax.bar(x, unresolved, bottom=resolved, color="#E7A75F", hatch="///", label="Reported remainder")
    ax.scatter(x, reported, color="#222222", marker="_", s=160, label="Source-reported total")
    ax.set_xticks(x, ["Ceramic filler", "Solar Salt"])
    ax.set_ylabel("50 K sensible inventory (MWh)")
    exponent_value = closure["filler_relative_error_pct"] * 1e3
    ax.text(0, reported[0]*1.06, f"error ${exponent_value:.2f} \\times 10^{{-3}}$%", ha="center", fontsize=6.8)
    ax.text(1, reported[1]*1.06, f"pore share {closure['geometric_pore_fraction_of_reported_salt_pct']:.2f}%", ha="center", fontsize=6.8)
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 2]; panel(ax, 5)
    # Confirmation is evaluated for every unit-design case.  Showing the joint
    # error distribution is more informative than a categorical evidence map
    # at this point in the argument: it establishes that the screen preserves
    # the two dynamic quantities used later in the design narrative.
    for orientation, marker, colour, label in [
        ("none", "o", "#3978A8", "Homogeneous"),
        ("fine_hot", "s", "#3E9B65", "Fine at hot end"),
        ("coarse_hot", "^", "#D75452", "Coarse at hot end"),
    ]:
        ids = t["codesign_results_v16"].loc[
            t["codesign_results_v16"].gradient_orientation.eq(orientation), "design_id"
        ]
        q = conf[conf.design_id.isin(ids)]
        ax.scatter(
            q.charge_t90_h_relative_difference_pct.abs(),
            q.high_grade_discharge_fraction_relative_difference_pct.abs(),
            s=7, alpha=0.18, marker=marker, color=colour, edgecolors="none", label=label,
            rasterized=True,
        )
    ax.set(
        xlabel=r"Absolute relative charge-$t_{90}$ difference (%)",
        ylabel=r"|Relative deliverable-energy-fraction difference| (%)",
    )
    clean(ax); legend_above(ax, 3)

    return finish(fig, MAIN / "Figure_01_model_validation_and_control_volume", 6,
                  ["external_validation_trace_v16", "numerical_verification_v16",
                    "particle_model_hierarchy_v16", "reference_control_volume_closure_v16", "screen_to_confirmation_v16"])


def figure_2(t: dict[str, object]) -> dict[str, object]:
    unit = t["codesign_results_v16"]
    adv = t["advantage_transfer_v16"]
    pairwise = t["pairwise_scale_order_v16"]
    merged = adv.merge(unit[["base_design_id", "topology_id", "fluid_model_id", "porosity"]],
                       on=["base_design_id", "topology_id", "fluid_model_id"], how="left")
    fig, axs = composite_grid()
    labels = [FLUID_SHORT[x] for x in FLUID_ORDER]

    ax = axs[0, 0]; panel(ax, 0)
    # A material-property coordinate is deliberately used instead of a
    # single-metric bar: it exposes the intrinsic capacity--viscosity tradeoff
    # that must not be inferred from the later filled-unit or service result.
    duty_temperature_K = 525.0 + 273.15
    for fluid in FLUID_ORDER:
        prop = fluid_properties(fluid, duty_temperature_K)
        capacity = prop.density_kg_m3 * prop.cp_J_kgK * 50.0 / 3.6e6
        size = 48.0 + 200.0 * prop.thermal_conductivity_W_mK / max(
            fluid_properties(f, duty_temperature_K).thermal_conductivity_W_mK for f in FLUID_ORDER
        )
        ax.scatter(capacity, prop.viscosity_Pa_s, s=size, color=PALETTE[fluid],
                   edgecolor="white", linewidth=.65, zorder=3)
        ax.annotate(FLUID_ANNOTATION_SHORT[fluid], (capacity, prop.viscosity_Pa_s),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.2)
    ax.set(xlabel="Fluid sensible capacity, 50 K (kWh m$^{-3}$)",
           ylabel="Viscosity at 525 °C (Pa s)", yscale="log")
    clean(ax)

    ax = axs[0, 1]; panel(ax, 1)
    parts = [unit.loc[unit.fluid_model_id == f, "volumetric_capacity_MWh_m3"].to_numpy() for f in FLUID_ORDER]
    vp = ax.violinplot(parts, positions=np.arange(5), showmedians=True, widths=0.75)
    for body, f in zip(vp["bodies"], FLUID_ORDER):
        body.set_facecolor(PALETTE[f]); body.set_alpha(0.65); body.set_edgecolor("none")
    ax.set_xticks(np.arange(5), labels, rotation=28, ha="right")
    ax.set_ylabel("Filled-module capacity (MWh m$^{-3}$)")
    clean(ax)

    ax = axs[0, 2]; panel(ax, 2)
    for f in FLUID_ORDER[1:]:
        q = merged[merged.fluid_model_id == f]
        sample = q.iloc[::max(1, len(q)//600)]
        ax.scatter(sample.porosity, sample.ARI_material_to_unit, s=8, alpha=0.18,
                   color=PALETTE[f], label=FLUID_SHORT[f])
        b = q.groupby(pd.cut(q.porosity, 8), observed=True).agg(
            porosity=("porosity", "median"), retention=("ARI_material_to_unit", "median"))
        ax.plot(b.porosity, b.retention, color=PALETTE[f], lw=1.8)
    ax.set(xlabel="Skeleton porosity", ylabel="Filled-module / fluid contrast ratio")
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 0]; panel(ax, 3)
    stages = ["A_material_log", "A_unit_log", "A_dynamic_log", "A_system_log"]
    stage_labels = ["Intrinsic\nfluid", "Filled\nmodule", "Finite-rate\ndelivery", "Power–energy\nsystem"]
    for f in FLUID_ORDER[1:]:
        q = adv[(adv.fluid_model_id == f) & (adv.scenario_id == "S10MW_8H")]
        med = [100*np.expm1(q[s].median()) for s in stages]
        lo = [100*np.expm1(q[s].quantile(.25)) for s in stages]
        hi = [100*np.expm1(q[s].quantile(.75)) for s in stages]
        ax.plot(range(4), med, "o-", color=PALETTE[f], label=FLUID_SHORT[f])
        ax.fill_between(range(4), lo, hi, color=PALETTE[f], alpha=0.12)
    ax.axhline(0, color="#444444", lw=0.7)
    ax.set_xticks(range(4), stage_labels)
    ax.set_ylabel("Signed logarithmic contrast (%)")
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 1]; panel(ax, 4)
    positions = []; data = []; colours = []
    for i, f in enumerate(FLUID_ORDER[1:]):
        for j, sc in enumerate(["S10MW_8H", "S50MW_6H"]):
            positions.append(i*2.5+j*0.75)
            data.append(adv[(adv.fluid_model_id == f) & (adv.scenario_id == sc)].ARI_material_to_system)
            colours.append(SCENARIO_COLOR[sc])
    bp = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
    for box, c in zip(bp["boxes"], colours): box.set(facecolor=c, alpha=.55, edgecolor=c)
    for median in bp["medians"]: median.set(color="#222222")
    ax.set_xticks([i*2.5+.375 for i in range(4)], [FLUID_SHORT[f] for f in FLUID_ORDER[1:]], rotation=27, ha="right")
    ax.set_ylabel("System / fluid contrast ratio")
    ax.legend([mpl.patches.Patch(color=SCENARIO_COLOR[s], alpha=.55) for s in SCENARIO_COLOR],
              [SCENARIO_LABEL[s] for s in SCENARIO_COLOR], loc="lower center", bbox_to_anchor=(.5,1.015), ncol=2, frameon=False)
    clean(ax)

    ax = axs[1, 2]; panel(ax, 5)
    material_difference = 100*np.expm1(pairwise.material_log_difference.abs())
    aligned_system_log = np.sign(pairwise.material_log_difference)*pairwise.system_log_difference
    aligned_system_difference = 100*np.sign(aligned_system_log)*np.expm1(aligned_system_log.abs())
    ties = pairwise.pair_fate.eq("system_exact_tie")
    preserved = ~ties
    ax.scatter(material_difference[preserved], aligned_system_difference[preserved],
               s=5, alpha=.09, color="#3978A8", edgecolors="none", rasterized=True,
               label="Order preserved")
    ax.scatter(material_difference[ties], aligned_system_difference[ties],
               s=18, marker="|", linewidths=.9, color="#D75452", label="Same module count")
    limit = float(material_difference.max())*1.03
    ax.plot([0, limit], [0, limit], "--", color="#333333", lw=.8, label="Equal contrast")
    ax.axhline(0, color="#777777", lw=.65)
    ax.set(xlim=(0, limit), ylim=(-.03*limit, limit),
           xlabel="Absolute fluid-level contrast (%)",
           ylabel="Order-aligned system-level contrast (%)")
    clean(ax); legend_above(ax, 2)

    return finish(fig, MAIN / "Figure_02_cross_scale_difference_contraction", 6,
                  ["codesign_results_v16", "advantage_transfer_v16", "pairwise_scale_order_v16"])


def _response_grid(df: pd.DataFrame, orientation: str, metric: str) -> tuple[np.ndarray, list[float], list[float]]:
    q = df[(df.topology == "graded_two_layer") & (df.gradient_orientation == orientation)]
    piv = q.pivot_table(index="layer_fraction", columns="gradient_ratio", values=metric, aggfunc="median")
    return piv.to_numpy(), list(piv.index), list(piv.columns)


def figure_3(t: dict[str, object]) -> dict[str, object]:
    topo = t["topology_response_v16"].copy()
    topo["high_grade_change_relative_pct"] = 100.0 * (
        topo.high_grade_discharge_fraction / topo.reference_high_grade_discharge_fraction - 1.0
    )
    unit = t["codesign_results_v16"]
    inter = t["interaction_indices_v16"]
    fig, axs = composite_grid()
    all_hg = topo.loc[topo.topology == "graded_two_layer", "high_grade_change_relative_pct"]
    bound = max(abs(all_hg.quantile(.01)), abs(all_hg.quantile(.99)))

    for idx, orientation in enumerate(["fine_hot", "coarse_hot"]):
        ax = axs[0, idx]; panel(ax, idx)
        z, rows, cols = _response_grid(topo, orientation, "high_grade_change_relative_pct")
        matrix(ax, z, [f"{x:.1f}" for x in rows], [f"{x:.1f}" for x in cols], "RdBu_r",
               -bound, bound, fmt=".2f", cbar_label="Relative deliverable-energy-fraction change (%)", center=0)
        ax.set(xlabel="Particle diameter ratio", ylabel="Hot-side layer fraction")

    ax = axs[0, 2]; panel(ax, 2)
    topo = topo.copy()
    topo["absolute_pressure_drop_change_Pa"] = 1000.0 * (
        topo["pressure_drop_kPa"] - topo["reference_pressure_drop_kPa"]
    )
    z, rows, cols = _response_grid(topo, "fine_hot", "absolute_pressure_drop_change_Pa")
    matrix(ax, z, [f"{x:.1f}" for x in rows], [f"{x:.1f}" for x in cols], "YlOrRd",
           float(np.nanmin(z)), float(np.nanmax(z)), fmt=".2f", cbar_label="Bed pressure-drop change (Pa)")
    ax.set(xlabel="Particle diameter ratio", ylabel="Hot-side layer fraction")

    ax = axs[1, 0]; panel(ax, 3)
    q = topo[topo.topology == "graded_two_layer"].iloc[::4]
    for orient in ["fine_hot", "coarse_hot"]:
        p = q[q.gradient_orientation == orient]
        sc = ax.scatter(p.absolute_pressure_drop_change_Pa, p.high_grade_change_relative_pct,
                        c=p.layer_fraction, cmap="viridis", vmin=.3, vmax=.7, s=12, alpha=.35,
                        marker="o" if orient == "fine_hot" else "^", label=orient.replace("_", " "))
    ax.axhline(0, color="#333333", lw=.7)
    ax.set(xlabel="Bed pressure-drop change (Pa)", ylabel="Relative deliverable-energy-fraction change (%)")
    ax.figure.colorbar(sc, ax=ax, fraction=.046, pad=.025, label="Hot-side layer fraction")
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 1]; panel(ax, 4)
    for orient in ["none", "fine_hot", "coarse_hot"]:
        p = unit[unit.gradient_orientation == orient].iloc[::3]
        sc = ax.scatter(p.particle_Biot_raw_max, p.thermal_front_thickness_ratio,
                        c=p.high_grade_discharge_fraction, cmap="viridis", vmin=.82, vmax=.95,
                        s=11, alpha=.30, marker={"none":"o","fine_hot":"s","coarse_hot":"^"}[orient],
                        label={"none":"homogeneous","fine_hot":"fine at hot end","coarse_hot":"coarse at hot end"}[orient])
    ax.set(xlabel="Maximum particle Biot number", ylabel="Front thickness / bed height")
    ax.figure.colorbar(sc, ax=ax, fraction=.046, pad=.025, label="Deliverable-energy fraction")
    clean(ax); legend_above(ax, 2)

    ax = axs[1, 2]; panel(ax, 5)
    short = {
        "porosity × particle_diameter_m": r"$\epsilon \times d_p$",
        "internal_diameter_m × particle_diameter_m": r"$D \times d_p$",
        "particle_diameter_m × nominal_thermal_input_kW": r"$d_p \times Q$",
        "internal_diameter_m × nominal_thermal_input_kW": "D × Q",
        "bed_height_m × particle_diameter_m": r"$H \times d_p$",
        "internal_diameter_m × porosity": "D × ε",
        "internal_diameter_m × bed_height_m": "D × H",
        "bed_height_m × nominal_thermal_input_kW": "H × Q",
    }
    top_names = inter.groupby("interaction").partial_R2.median().nlargest(6).index
    metrics = ["charge_t90_h", "high_grade_discharge_fraction", "pressure_drop_kPa"]
    piv = inter[inter.interaction.isin(top_names)].pivot_table(index="interaction", columns="metric", values="partial_R2", aggfunc="median").reindex(index=top_names, columns=metrics)
    matrix(ax, piv.to_numpy(), [short.get(x, x) for x in piv.index], ["Charge $t_{90}$", "Deliverable fraction", "Pressure drop"],
           "YlGnBu", 0, max(.05, float(np.nanmax(piv.to_numpy()))), fmt=".3f", cbar_label="Median incremental $R^2$")

    return finish(fig, MAIN / "Figure_03_structure_performance_tradeoffs", 6,
                  ["topology_response_v16", "codesign_results_v16", "interaction_indices_v16"])


def _pareto_panel(ax: plt.Axes, svc: pd.DataFrame, front: pd.DataFrame, scenario: str) -> None:
    q = svc[svc.scenario_id == scenario]
    ax.scatter(q.installed_packed_volume_m3, q.salt_inventory_t, s=5, color="#C9CED4", alpha=.18, rasterized=True)
    f = front[front.scenario_id == scenario]
    for fluid in FLUID_ORDER:
        p = f[f.fluid_model_id == fluid]
        if p.empty: continue
        markers = ["s" if x == "fine_hot" else "o" for x in p.gradient_orientation]
        for (_, row), marker in zip(p.iterrows(), markers):
            ax.scatter(row.installed_packed_volume_m3, row.salt_inventory_t,
                       s=42, marker=marker,
                       color=PALETTE[fluid], edgecolor="white", lw=.6, zorder=5)
    ax.set(xlabel="Installed packed volume (m$^3$)", ylabel="Salt inventory (t)")
    clean(ax)


def figure_4(t: dict[str, object]) -> dict[str, object]:
    # Service closure tables intentionally carry only service-level fields.
    # Join the registered vessel diameter once for the header-friction panel;
    # this is a traceable display join, not a recalculation of any objective.
    unit_meta = t["codesign_results_v16"][["design_id", "internal_diameter_m"]]
    svc = t["fixed_service_results_v16"].merge(unit_meta, on="design_id", how="left", validate="many_to_one")
    front = t["system_pareto_front_v16"].merge(unit_meta, on="design_id", how="left", validate="many_to_one")
    fig, axs = composite_grid()

    ax = axs[0, 0]; panel(ax, 0)
    positions=[]; data=[]; colours=[]
    for i, f in enumerate(FLUID_ORDER):
        for j, sc in enumerate(SCENARIO_COLOR):
            positions.append(i*2.3+j*.7); data.append(svc[(svc.fluid_model_id==f)&(svc.scenario_id==sc)].resolved_to_nominal_unit_duty_ratio); colours.append(SCENARIO_COLOR[sc])
    bp=ax.boxplot(data,positions=positions,widths=.52,showfliers=False,patch_artist=True)
    for b,c in zip(bp['boxes'],colours): b.set(facecolor=c,alpha=.52,edgecolor=c)
    for m in bp['medians']:m.set(color='#222222')
    ax.set_xticks([i*2.3+.35 for i in range(5)],[FLUID_SHORT[f] for f in FLUID_ORDER],rotation=27,ha='right')
    ax.set_ylabel("Required / nominal module power")
    ax.legend([mpl.patches.Patch(color=SCENARIO_COLOR[s],alpha=.52) for s in SCENARIO_COLOR],[SCENARIO_LABEL[s] for s in SCENARIO_COLOR],loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2,frameon=False)
    clean(ax)

    ax=axs[0,1]; panel(ax,1)
    for sc in SCENARIO_COLOR:
        q=svc[svc.scenario_id==sc].iloc[::4]
        ax.scatter(q.nominal_high_grade_discharge_fraction,q.resolved_high_grade_discharge_fraction,s=8,alpha=.18,color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc])
    lo=min(svc.nominal_high_grade_discharge_fraction.min(),svc.resolved_high_grade_discharge_fraction.min()); hi=max(svc.nominal_high_grade_discharge_fraction.max(),svc.resolved_high_grade_discharge_fraction.max())
    ax.plot([lo,hi],[lo,hi],'--',color='#333333',lw=.8,label='Equal to nominal-flow result')
    ax.set(xlabel="Nominal-flow deliverable-energy fraction",ylabel="Recalculated deliverable-energy fraction")
    clean(ax);legend_above(ax,2)

    ax=axs[0,2]; panel(ax,2)
    for sc in SCENARIO_COLOR:
        q=np.sort(svc.loc[svc.scenario_id==sc,'capacity_oversize_pct'].to_numpy())
        ax.plot(q,100*np.arange(1,len(q)+1)/len(q),color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc])
        pf=front[front.scenario_id==sc]
        ax.scatter(pf.capacity_oversize_pct,np.interp(pf.capacity_oversize_pct,q,100*np.arange(1,len(q)+1)/len(q)),color=SCENARIO_COLOR[sc],edgecolor='white',s=30,zorder=4)
    ax.set(xscale='symlog',xlabel="Excess capacity (%)",ylabel="Cumulative designs (%)")
    clean(ax);legend_above(ax,2)

    ax=axs[1,0];panel(ax,3);_pareto_panel(ax,svc,front,'S10MW_8H')
    ax=axs[1,1];panel(ax,4);_pareto_panel(ax,svc,front,'S50MW_6H')
    handles=[Line2D([0],[0],marker='o',ls='',color=PALETTE[f],label=FLUID_SHORT[f],markersize=5) for f in FLUID_ORDER if f in set(front.fluid_model_id)]
    axs[1,0].legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.015),ncol=min(3,len(handles)),frameon=False)
    axs[1,1].legend(handles=[Line2D([0],[0],marker='o',ls='',color='#666',label='Homogeneous',markersize=5),Line2D([0],[0],marker='s',ls='',color='#666',label='Fine at hot end',markersize=5)],loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2,frameon=False)

    ax=axs[1,2];panel(ax,5)
    q=svc.iloc[::4].copy()
    diameter_scatter = None
    for sc, marker in [("S10MW_8H", "o"), ("S50MW_6H", "s")]:
        p=q[q.scenario_id==sc]
        diameter_scatter=ax.scatter(p.parallel_unit_count,p.header_pressure_drop_kPa,
                                    c=p.internal_diameter_m,cmap='viridis',vmin=.65,vmax=1.15,
                                    marker=marker,s=13,alpha=.28,edgecolors='none',label=SCENARIO_LABEL[sc])
        f=front[front.scenario_id==sc]
        ax.scatter(f.parallel_unit_count,f.header_pressure_drop_kPa,
                   c=f.internal_diameter_m,cmap='viridis',vmin=.65,vmax=1.15,
                   marker=marker,s=42,edgecolors='white',linewidth=.6,zorder=4)
    ax.set(xscale='log',xlabel='Required module count',ylabel='Header pressure drop (kPa)')
    clean(ax);legend_above(ax,2)
    cb=ax.figure.colorbar(diameter_scatter,ax=ax,fraction=.046,pad=.025)
    cb.set_label('Module diameter (m)')

    return finish(fig, MAIN / "Figure_04_deliverable_service_and_pareto", 6,
                  ["fixed_service_results_v16", "system_pareto_front_v16"])


def figure_5(t: dict[str, object]) -> dict[str, object]:
    sk=t['skeleton_within_source_effects']; nano=t['nanoparticle_dose_response'];ev=t['evidence_coverage'];arch=t['architecture_gate'];pair=t['pair_gate']
    fig, axs = composite_grid(6.9)
    ax=axs[0,0];panel(ax,0)
    for _,r in sk.iterrows():
        ax.scatter(r.capacity_change_pct_vs_matched_baseline,r.rate_or_transport_change_pct_vs_matched_baseline,s=55,color='#3978A8' if r.cross_architecture_rank_eligible else '#E6862A',edgecolor='white',zorder=3)
        ax.annotate(candidate_short(r.candidate_id),(r.capacity_change_pct_vs_matched_baseline,r.rate_or_transport_change_pct_vs_matched_baseline),xytext=(4,3),textcoords='offset points',fontsize=6.5)
    ax.axvline(0,color='#555',lw=.7);ax.axhline(0,color='#555',lw=.7)
    ax.set(xlabel="Within-study capacity metric change (%)",ylabel="Within-study rate or transport change (%)")
    clean(ax)

    ax=axs[0,1];panel(ax,1)
    q=nano[nano.nanoparticle!='none'].copy()
    for (group,npname),p in q.groupby(['study_group','nanoparticle']):
        p=p.sort_values('loading_wt_pct'); label=nanoparticle_short(npname)
        ax.plot(p.loading_wt_pct,p.reported_cp_change_pct,'o-',label=label)
    ax.axhline(0,color='#444',lw=.7)
    ax.set(xlabel="Loading (wt%)",ylabel="$c_p$ change (%)")
    clean(ax)
    ax.legend(loc='lower center',bbox_to_anchor=(.68,1.015),ncol=3,frameon=False,
              columnspacing=.60,handletextpad=.30,borderaxespad=0.0,fontsize=6.5)

    ax=axs[0,2];panel(ax,2)
    maxima=q.loc[q.groupby(['study_group','nanoparticle']).reported_cp_change_pct.idxmax()].copy()
    x=np.arange(len(maxima))
    ax.bar(x,maxima.reported_cp_change_pct,color=plt.cm.Set2(np.linspace(0,1,len(maxima))))
    ax2=ax.twinx()
    ax2.scatter(x,maxima.loading_wt_pct,color='#222222',marker='D',s=24,label='Loading at max.')
    ax2.set_ylabel('Loading at maximum (wt%)')
    ax2.set_ylim(0,max(3.0,float(maxima.loading_wt_pct.max())*1.25))
    ax2.spines['top'].set_visible(False)
    maxima_labels=[
        'GNP' if str(name)=='graphene nanoplatelets' else nanoparticle_short(name)
        for name in maxima.nanoparticle
    ]
    ax.set_xticks(x,maxima_labels,rotation=22,ha='right')
    ax.set_ylabel("Maximum $c_p$ change (%)")
    clean(ax)
    ax2.legend(loc='lower center',bbox_to_anchor=(.5,1.015),frameon=False)

    ax=axs[1,0];panel(ax,3)
    ranges=[]
    import re
    for _,row in sk.iterrows():
        nums=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',str(row.temperature_window_C))]
        if len(nums)>=2:lo,hi=nums[0],nums[1]
        elif len(nums)==1:lo=hi=nums[0]
        else:lo=hi=np.nan
        ranges.append((candidate_short(row.candidate_id),lo,hi))
    for i,(name,lo,hi) in enumerate(ranges):
        if lo==hi:ax.scatter(lo,i,s=45,color='#E6862A')
        else:ax.plot([lo,hi],[i,i],lw=7,color='#3978A8',solid_capstyle='butt')
    ax.axvspan(500,550,color='#BFC5CB',alpha=.28,label='Common study range')
    ax.set_yticks(range(len(ranges)),[x[0] for x in ranges]);ax.set_xlabel('Measurement window (\u00b0C)');clean(ax);legend_above(ax,1)

    ax=axs[1,1];panel(ax,4)
    y=np.arange(len(arch));col=['#3E9B65' if x else '#BFC5CB' for x in arch.dynamic_common_rank_eligible]
    ax.barh(y,arch.missing_evidence_count,color=col)
    ax.scatter(np.where(arch.within_source_quantitation,arch.missing_evidence_count+.15,np.nan),y,marker='D',color='#E6862A',label='Within-study quantification')
    ax.set_yticks(y,[architecture_short(x) for x in arch.architecture_id]);ax.invert_yaxis();ax.set_xlabel("Unavailable transient-model inputs")
    clean(ax);legend_above(ax,1)

    ax=axs[1,2];panel(ax,5)
    p=pair.set_index('fluid_model_id').loc[FLUID_ORDER].reset_index()
    cols=['temperature_overlap_status','chemical_compatibility_status','wetting_status','cycle_stability_status','engineering_deployment_eligible','evidence_tier']
    arr=np.zeros((len(p),len(cols)));txt=[]
    for _,r in p.iterrows():
        row=[]
        for j,c in enumerate(cols):
            s=str(r[c]).lower()
            if c=='engineering_deployment_eligible': v=2 if s=='true' else 0; lab='yes' if v else 'no'
            elif c=='evidence_tier':
                v=2 if 'unit_pair' in s else 1
                lab='pair' if v==2 else 'common'
            elif 'direct' in s or 'resolved_for' in s: v=2;lab='direct'
            elif 'source_specific' in s or 'partial' in s: v=1;lab='partial'
            else:v=0;lab='open'
            arr[len(txt),j]=v;row.append(lab)
        txt.append(row)
    matrix(ax,arr,[FLUID_SHORT[x] for x in FLUID_ORDER],['Thermal','Chemical','Wetting','Cycling','Complete','Scope'],ListedColormap(['#E5E7EA','#F2C35D','#4FAF9F']),0,2,text_values=txt,text_fontsize=5.9)
    ax.tick_params(axis='x', labelsize=6.6)

    return finish(fig,MAIN/'Figure_05_skeleton_and_nanoparticle_evidence',6,
                  ['skeleton_within_source_effects','nanoparticle_dose_response','evidence_coverage','architecture_gate','salt_skeleton_pair_registry'])


def figure_6(t: dict[str, object]) -> dict[str, object]:
    prec=t['pareto_precision_sensitivity_v16'];samp=t['sampling_robustness_v16'];rob=t['bounded_robustness_v16'];ms=t['model_form_stress_v16'];stand=t['standby_loss_v16']
    fig, axs = composite_grid()
    ax=axs[0,0];panel(ax,0)
    for sc in SCENARIO_COLOR:
        q=prec[prec.scenario_id==sc].sort_values('decision_precision_multiplier')
        ax.plot(q.decision_precision_multiplier,q.n_front_designs,'o-',color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc])
    ax.set(xlabel="Objective-resolution threshold multiplier",ylabel="Nondominated designs",yscale='log')
    clean(ax);legend_above(ax,2)

    for idx,metric,ylabel in [(1,'IGD_plus','Inverted generational distance plus'),(2,'additive_epsilon_indicator','Additive epsilon indicator')]:
        ax=axs[0,idx];panel(ax,idx)
        for sc,off in [('S10MW_8H',-.08),('S50MW_6H',.08)]:
            q=samp[samp.scenario_id==sc]
            for seed,p in q.groupby('seed'):
                ax.plot(p.n_base_geometries+off*100,p[metric],color=SCENARIO_COLOR[sc],alpha=.25,lw=.7)
            g=q.groupby('n_base_geometries')[metric].agg(['median','min','max'])
            ax.errorbar(g.index+off*100,g['median'],yerr=[g['median']-g['min'],g['max']-g['median']],fmt='o-',capsize=2,color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc])
        ax.set(xlabel="Geometries per replicate",ylabel=ylabel);ax.set_xticks([64,128,192])
        clean(ax);legend_above(ax,2)

    ax=axs[1,0];panel(ax,3)
    assessed=rob[rob.n_stress_samples>0]
    for sc in SCENARIO_COLOR:
        q=assessed[assessed.scenario_id==sc]
        ax.scatter(q.median_equal_weight_regret_pct,q.pareto_attainment_frequency_pct,s=40,color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc],alpha=.8)
    ax.set(xlabel="Median regret (%)",ylabel="Nondominated-set attainment (%)")
    clean(ax);legend_above(ax,2)

    ax=axs[1,1];panel(ax,4)
    base=ms[ms.stress_id=='intrinsic_adiabatic'].set_index(['scenario_id','design_id'])
    rows=[]
    for stid,label,c in [('bounded_moderate','Moderate','#E6862A'),('bounded_severe','Severe','#D75452')]:
        p=ms[ms.stress_id==stid].set_index(['scenario_id','design_id'])
        changes=[100*(p.installed_packed_volume_m3/base.installed_packed_volume_m3-1),100*(p.resolved_high_grade_discharge_fraction/base.resolved_high_grade_discharge_fraction-1),p.charge_discharge_t90_asymmetry_pct]
        for j,v in enumerate(changes):
            v=v[np.isfinite(v)]
            if v.empty:
                continue
            ax.scatter(v,np.full(len(v),j)+(-.08 if stid=='bounded_moderate' else .08),s=12,alpha=.3,color=c)
            ax.plot([v.min(),v.max()],[j+(-.08 if stid=='bounded_moderate' else .08)]*2,color=c,lw=2)
            ax.scatter(v.median(),j+(-.08 if stid=='bounded_moderate' else .08),s=38,color=c,edgecolor='white',zorder=4,label=label if j==0 else None)
    ax.axvline(0,color='#444',lw=.7);ax.set_yticks(range(3),['Volume change (%)','Deliverable-energy\nfraction change (%)','$t_{90}$ asymmetry (%)'])
    ax.set_xlabel("Relative change or asymmetry (%)")
    clean(ax);legend_above(ax,2)

    ax=axs[1,2];panel(ax,5)
    for u,c in [(0.1,'#3978A8'),(0.5,'#D75452')]:
        q=stand[stand.surface_heat_transfer_coefficient_W_m2K==u]
        g=q.groupby('standby_duration_h').standby_sensible_loss_pct.agg(['median','min','max'])
        ax.plot(g.index,g['median'],'o-',color=c,label=f"U = {u:.2f} W m$^{{-2}}$ K$^{{-1}}$")
        ax.fill_between(g.index,g['min'],g['max'],color=c,alpha=.15)
    ax.set(xlabel="Standby duration (h)",ylabel="Inventory loss (%)")
    clean(ax);legend_above(ax,1)

    return finish(fig,MAIN/'Figure_06_robustness_and_loss_bounds',6,
                  ['pareto_precision_sensitivity_v16','sampling_robustness_v16','bounded_robustness_v16','model_form_stress_v16','standby_loss_v16'])


def supplementary_1(t: dict[str, object]) -> dict[str, object]:
    fig, axs = composite_grid()
    temps=np.linspace(500,550,51)+273.15
    metrics=[('density_kg_m3','Density (kg m$^{-3}$)'),('cp_J_kgK','$c_p$ (J kg$^{-1}$ K$^{-1}$)'),('viscosity_Pa_s','Viscosity (Pa s)'),('thermal_conductivity_W_mK','Conductivity (W m$^{-1}$ K$^{-1}$)')]
    for idx,(attr,label) in enumerate(metrics):
        ax=axs.flat[idx];panel(ax,idx)
        for f in FLUID_ORDER:
            ax.plot(temps-273.15,[getattr(fluid_properties(f,float(T)),attr) for T in temps],color=PALETTE[f],label=FLUID_SHORT[f])
        ax.set(xlabel='Temperature (°C)',ylabel=label);clean(ax)
    ax=axs.flat[4];panel(ax,4)
    for i,f in enumerate(FLUID_ORDER):
        spec=get_model_spec(f);ax.plot([spec.valid_temperature_min_C,spec.valid_temperature_max_C],[i,i],lw=7,color=PALETTE[f])
    ax.axvspan(500,550,color='#BFC5CB',alpha=.28,label='Common temperature range')
    ax.set_yticks(range(5),[FLUID_SHORT[f] for f in FLUID_ORDER]);ax.set_xlabel('Temperature domain (°C)');clean(ax);legend_above(ax,1)
    ax=axs.flat[5];panel(ax,5)
    unit=t['codesign_results_v16'];v=unit.groupby('fluid_model_id')[['fluid_volumetric_sensible_kWh_m3','Pr','Nu_min','Nu_max']].median().reindex(FLUID_ORDER)
    z=(v-v.min())/(v.max()-v.min())
    matrix(ax,z.to_numpy(),[FLUID_SHORT[f] for f in FLUID_ORDER],['Fluid capacity','Prandtl','Nu min','Nu max'],'Blues',0,1,fmt='.2f',cbar_label='Normalized median')
    figure_legend(fig, [Line2D([0], [0], color=PALETTE[f]) for f in FLUID_ORDER],
                  [FLUID_ANNOTATION_SHORT[f] for f in FLUID_ORDER], ncol=3)
    return finish(fig,SUPP/'Supplementary_Figure_S01_source_qualified_fluid_properties',6,['fluid_properties.py','codesign_results_v16'])


def supplementary_2(t: dict[str, object]) -> dict[str, object]:
    nv=t['numerical_verification_v16'];refine=t['decision_grid_refinement_v16'];fig, axs = composite_grid(ncols=2)
    grid=nv[nv.verification_case=='grid_energy_convergence']
    for idx,(m,l) in enumerate([('front_thickness_ratio','Front thickness / bed height'),('charge_energy_balance_error_pct','Charge balance error (%)')]):
        ax=axs.flat[idx];panel(ax,idx)
        for topo,c in [('homogeneous','#3978A8'),('graded_two_layer','#3E9B65')]:
            q=grid[grid.topology==topo];ax.plot(q.n_cells,q[m].abs() if 'error' in m else q[m],'o-',color=c,label=topo.replace('_',' '))
        ax.set(xscale='log',xlabel='Axial cells',ylabel=l);ax.set_xticks([64,128,256,512,1024,2048,4096],[64,128,256,512,1024,2048,4096]);clean(ax)
        if idx==0:legend_above(ax,2)
    ax=axs.flat[2];panel(ax,2)
    adv=nv[nv.verification_case=='periodic_step_advection']
    method_labels={
        'first_order_upwind':'First-order upwind',
        'MUSCL_minmod':'Second-order minmod reconstruction',
    }
    for method,c in [('first_order_upwind','#D75452'),('MUSCL_minmod','#3E9B65')]:
        q=adv[adv.method==method]
        ax.plot(q.transition_cells_0p1_to_0p9,q.L1_error,'o-',color=c,label=method_labels[method])
    ax.set(xlabel='Transition width (cells)',ylabel=r'Normalized spatial $L_1$ error');clean(ax);legend_above(ax,2)
    ax=axs.flat[3];panel(ax,3)
    changes=[]
    for i,sc in enumerate(SCENARIO_COLOR):
        q=refine.loc[refine.scenario_id==sc,'deliverable_energy_change_pct'].abs()
        normalized=100.0*q/0.5
        changes.append(normalized.to_numpy())
        jitter=np.linspace(-.08,.08,len(normalized))
        ax.scatter(np.full(len(normalized),i)+jitter,normalized,color=SCENARIO_COLOR[sc],s=28,
                   edgecolor='white',lw=.45,zorder=3)
    bp=ax.boxplot(changes,positions=np.arange(2),widths=.52,showfliers=False,patch_artist=True)
    for box,c in zip(bp['boxes'],SCENARIO_COLOR.values()): box.set_facecolor(c);box.set_alpha(.22);box.set_edgecolor(c)
    for median in bp['medians']: median.set_color('#222222')
    ax.axhline(100,color='#555',lw=.75,ls=':')
    ax.set_xticks(np.arange(2),[SCENARIO_LABEL[x] for x in SCENARIO_COLOR])
    ax.set(xlabel='Power–energy requirement',ylabel=r'|$\Delta$ deliverable energy| / 0.5% limit (%)',ylim=(0,108))
    clean(ax)
    return finish(fig,SUPP/'Supplementary_Figure_S02_numerical_verification',4,['numerical_verification_v16','decision_grid_refinement_v16'])


def supplementary_3(t: dict[str, object]) -> dict[str, object]:
    g=t['geometry_pool'].copy()
    g['storage_volume_m3']=math.pi*g.internal_diameter_m**2*g.bed_height_m/4.0
    g['particle_diameter_min_m']=g.particle_diameter_m/np.sqrt(g.gradient_ratio)
    base=g[g.gradient_orientation=='none'].copy()
    fig, axs = composite_grid()
    pairs=[
        ('internal_diameter_m','bed_height_m','nominal_thermal_input_kW','Nominal thermal input (kW)','Vessel diameter, D (m)','Bed height, H (m)'),
        ('porosity','particle_diameter_m','nominal_thermal_input_kW','Nominal thermal input (kW)','Porosity','Particle diameter, $d_p$ (m)'),
        ('nominal_thermal_input_kW','storage_volume_m3','porosity','Porosity','Nominal thermal input (kW)','Packed volume (m$^3$)'),
    ]
    for idx,(x,y,c,label,xlabel,ylabel) in enumerate(pairs):
        ax=axs.flat[idx];panel(ax,idx)
        sc=ax.scatter(base[x],base[y],c=base[c],s=11,alpha=.5,cmap='viridis',edgecolor='none')
        ax.figure.colorbar(sc,ax=ax,fraction=.046,pad=.025,label=label)
        ax.set(xlabel=xlabel,ylabel=ylabel);clean(ax)
    ax=axs.flat[3];panel(ax,3)
    q=g[g.gradient_orientation=='fine_hot']
    ax.scatter(q.particle_diameter_m,q.particle_diameter_min_m,s=10,alpha=.35,color='#3978A8',label='Fine layer')
    ax.scatter(q.particle_diameter_m,q.particle_diameter_m*np.sqrt(q.gradient_ratio),s=10,alpha=.35,color='#D75452',label='Coarse layer')
    lo=float(g.particle_diameter_m.min());hi=float(g.particle_diameter_m.max())
    ax.plot([lo,hi],[lo,hi],'--',color='#555',lw=.7)
    ax.set(xlabel='Base particle diameter (m)',ylabel='Layer diameter (m)');clean(ax);legend_above(ax,2)
    ax=axs.flat[4];panel(ax,4)
    variables=['internal_diameter_m','bed_height_m','porosity','particle_diameter_m','nominal_thermal_input_kW']
    z=g[g.gradient_orientation=='none'][variables].copy();z=(z-z.min())/(z.max()-z.min())
    vp=ax.violinplot([z[c] for c in variables],positions=np.arange(5),showmedians=True,widths=.72)
    for body in vp['bodies']:body.set_facecolor('#3E9B65');body.set_alpha(.55);body.set_edgecolor('none')
    ax.set_xticks(np.arange(5),['D','H','Porosity','$d_p$','Thermal input']);ax.set_ylabel('Normalized design range');clean(ax)
    ax=axs.flat[5];panel(ax,5)
    cols=['internal_diameter_m','bed_height_m','porosity','particle_diameter_m','nominal_thermal_input_kW']
    corr=g[g.gradient_orientation=='none'][cols].corr().to_numpy();matrix(ax,corr,['D','H','Porosity','$d_p$','Thermal input'],['D','H','Porosity','$d_p$','Thermal input'],'RdBu_r',-1,1,fmt='.2f',center=0,cbar_label='Pearson r')
    return finish(fig,SUPP/'Supplementary_Figure_S03_design_space_coverage',6,['geometry_pool'])


def supplementary_4(t: dict[str, object]) -> dict[str, object]:
    u=t['codesign_results_v16'];fig, axs = composite_grid()
    metrics=[('volumetric_capacity_MWh_m3','Capacity (MWh m$^{-3}$)'),('high_grade_discharge_fraction','Deliverable-energy fraction'),('pressure_drop_kPa','Pressure drop (kPa)'),('charge_t90_h','Charge $t_{90}$ (h)')]
    ax=axs.flat[0];panel(ax,0)
    sample=u.iloc[::3]
    for f in FLUID_ORDER:
        q=sample[sample.fluid_model_id==f];ax.scatter(q.salt_inventory_kg,q.solid_inventory_kg,s=8,alpha=.22,color=PALETTE[f],label=FLUID_SHORT[f])
    ax.set(xlabel='Module salt inventory (kg)',ylabel='Module solid inventory (kg)');clean(ax);legend_above(ax,3)
    for idx,(m,l) in enumerate(metrics[1:],start=1):
        ax=axs.flat[idx];panel(ax,idx)
        for f in FLUID_ORDER:
            q=np.sort(u.loc[u.fluid_model_id==f,m]);ax.plot(q,100*np.arange(1,len(q)+1)/len(q),color=PALETTE[f],label=FLUID_SHORT[f])
        ax.set(xlabel=l,ylabel='Cumulative single-module designs (%)');clean(ax)
    ax=axs.flat[4];panel(ax,4)
    q=u.iloc[::3];sc=ax.scatter(q.pump_energy_fraction,q.high_grade_discharge_fraction,c=q.porosity,cmap='viridis',s=9,alpha=.3);ax.figure.colorbar(sc,ax=ax,fraction=.046,pad=.025,label='Porosity');ax.set(xlabel='Frictional hydraulic-energy fraction',ylabel='Deliverable-energy fraction');clean(ax)
    ax=axs.flat[5];panel(ax,5)
    q=u.iloc[::3];ax.scatter(q.salt_inventory_kg,q.volumetric_capacity_MWh_m3,c=q.storage_volume_m3,cmap='plasma',s=9,alpha=.3);ax.set(xlabel='Module salt inventory (kg)',ylabel='Capacity (MWh m$^{-3}$)');clean(ax)
    return finish(fig,SUPP/'Supplementary_Figure_S04_unit_response_distributions',6,['codesign_results_v16'])


def supplementary_5(t: dict[str, object]) -> dict[str, object]:
    """Disaggregate topology without five repeated fluid-specific line charts."""
    d=t['topology_response_v16'].copy();fig, axs = composite_grid(ncols=2)
    d['high_grade_change_relative_pct']=100.0*(d.high_grade_discharge_fraction/d.reference_high_grade_discharge_fraction-1.0)
    fractions=sorted(d.layer_fraction.unique())
    frac_colours=plt.cm.viridis(np.linspace(.14,.86,len(fractions)))
    for idx,orientation in enumerate(['fine_hot','coarse_hot']):
        ax=axs[0,idx];panel(ax,idx)
        q=d[(d.topology=='graded_two_layer')&(d.gradient_orientation==orientation)]
        for frac,c in zip(fractions,frac_colours):
            g=q[q.layer_fraction==frac].groupby('gradient_ratio').high_grade_change_relative_pct.agg(['median','min','max']).sort_index()
            ax.fill_between(g.index,g['min'],g['max'],color=c,alpha=.10,linewidth=0)
            ax.plot(g.index,g['median'],'o-',color=c,ms=3.0)
        ax.axhline(0,color='#444',lw=.7)
        ax.set(xlabel='Particle diameter ratio',ylabel='Relative deliverable-energy-fraction change (%)')
        clean(ax)
    ax=axs[1,0];panel(ax,2)
    q=d[d.topology=='graded_two_layer'].groupby(['fluid_model_id','gradient_orientation']).high_grade_change_relative_pct.median().unstack().reindex(FLUID_ORDER)
    bound=max(abs(float(q.min().min())),abs(float(q.max().max())))
    matrix(ax,q.to_numpy(),[FLUID_SHORT[f] for f in FLUID_ORDER],['Fine hot','Coarse hot'],'RdBu_r',-bound,bound,fmt='.2f',cbar_label='Median relative deliverable-energy-fraction change (%)',center=0)
    ax.set_xlabel('Grading orientation')
    ax=axs[1,1];panel(ax,3)
    d['absolute_pressure_drop_change_Pa']=1000.0*(d.pressure_drop_kPa-d.reference_pressure_drop_kPa)
    q=d[d.topology=='graded_two_layer'].groupby(['fluid_model_id','gradient_orientation']).agg(hg=('high_grade_change_relative_pct','median'),dp=('absolute_pressure_drop_change_Pa','median')).reset_index()
    bound=max(abs(float(q.hg.min())),abs(float(q.hg.max())))
    norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    last=None
    for o,m,offset in [('fine_hot','o',-0.11),('coarse_hot','^',0.11)]:
        p=q[q.gradient_orientation==o].set_index('fluid_model_id').loc[FLUID_ORDER].reset_index()
        last=ax.scatter(p.dp,np.arange(len(p))+offset,c=p.hg,cmap='RdBu_r',norm=norm,marker=m,s=54,
                        edgecolor='white',linewidth=.55,label=o.replace('_',' '),zorder=3)
    ax.axvline(0,color='#444',lw=.7)
    ax.set_yticks(np.arange(len(FLUID_ORDER)),[FLUID_SHORT[f] for f in FLUID_ORDER])
    ax.set(xlabel='Median bed pressure-drop change (Pa)',ylabel='Fluid')
    ax.figure.colorbar(last,ax=ax,fraction=.046,pad=.025,label='Median relative deliverable-energy-fraction change (%)')
    clean(ax);legend_above(ax,2)
    figure_legend(fig,[Line2D([0],[0],color=c,marker='o',lw=1.25,ms=3) for c in frac_colours],
                  [f'Hot-side fraction {x:.1f}' for x in fractions],ncol=3)
    return finish(fig,SUPP/'Supplementary_Figure_S05_topology_fluid_disaggregation',4,['topology_response_v16'])


def supplementary_6(t: dict[str, object]) -> dict[str, object]:
    s=t['fixed_service_results_v16'];f=t['system_pareto_front_v16'];p=t['pareto_precision_sensitivity_v16'];fig, axs = composite_grid()
    panels=[('installed_packed_volume_m3','pump_energy_fraction'),('salt_inventory_t','capacity_oversize_pct')]
    axis_label={
        'installed_packed_volume_m3':'Installed volume (m$^3$)',
        'pump_energy_fraction':'Hydraulic energy fraction',
        'salt_inventory_t':'Salt inventory (t)',
        'capacity_oversize_pct':'Excess capacity (%)',
    }
    idx=0
    for sc in SCENARIO_COLOR:
        for x,y in panels:
            ax=axs.flat[idx];panel(ax,idx);q=s[s.scenario_id==sc].iloc[::2];ax.scatter(q[x],q[y],s=6,color='#C9CED4',alpha=.2);ff=f[f.scenario_id==sc];ax.scatter(ff[x],ff[y],c=[PALETTE[z] for z in ff.fluid_model_id],s=38,edgecolor='white');ax.set(xlabel=axis_label[x],ylabel=axis_label[y]);clean(ax);idx+=1
    ax=axs.flat[4];panel(ax,4)
    for sc in SCENARIO_COLOR:
        q=p[p.scenario_id==sc]
        ax.plot(q.decision_precision_multiplier,q.n_fluids,'o-',label=f"Fluids, {SCENARIO_LABEL[sc]}",color=SCENARIO_COLOR[sc])
        ax.plot(q.decision_precision_multiplier,q.n_topologies,'s--',label=f"Topologies, {SCENARIO_LABEL[sc]}",color=SCENARIO_COLOR[sc])
    ax.set(xlabel='Objective-resolution threshold multiplier',ylabel='Distinct material or arrangement identities');clean(ax);legend_above(ax,2)
    ax=axs.flat[5];panel(ax,5)
    for sc in SCENARIO_COLOR:
        q=np.sort(s.loc[s.scenario_id==sc,'parallel_unit_count']);ax.plot(q,100*np.arange(1,len(q)+1)/len(q),label=SCENARIO_LABEL[sc],color=SCENARIO_COLOR[sc])
    ax.set(xlabel='Required module count',ylabel='Cumulative designs (%)');clean(ax);legend_above(ax,2)
    return finish(fig,SUPP/'Supplementary_Figure_S06_service_objective_projections',6,['fixed_service_results_v16','system_pareto_front_v16','pareto_precision_sensitivity_v16'])


def supplementary_7(t: dict[str, object]) -> dict[str, object]:
    s=t['sampling_robustness_v16'];r=t['bounded_robustness_v16'];m=t['model_form_stress_v16'];fig, axs = composite_grid()
    idx=0
    for sc in SCENARIO_COLOR:
        for metric,label in [('IGD_plus','Inverted generational distance plus'),('additive_epsilon_indicator','Additive epsilon indicator')]:
            ax=axs.flat[idx];panel(ax,idx);q=s[s.scenario_id==sc]
            for seed,p in q.groupby('seed'):ax.plot(p.n_base_geometries,p[metric],'o-',label=str(seed))
            seed_labels={seed:f'Replicate {i + 1}' for i,seed in enumerate(sorted(q.seed.unique()))}
            ax.clear();panel(ax,idx)
            for seed,p in q.groupby('seed'):
                ax.plot(p.n_base_geometries,p[metric],'o-',label=seed_labels[seed])
            ax.set(xlabel='Geometries per replicate',ylabel=label);ax.set_xticks([64,128,192]);clean(ax)
            if idx==0:legend_above(ax,2)
            idx+=1
    ax=axs.flat[4];panel(ax,4)
    q=s.groupby(['scenario_id','n_base_geometries']).n_service_pareto.agg(['median','min','max']).reset_index()
    for sc in SCENARIO_COLOR:
        p=q[q.scenario_id==sc];ax.errorbar(p.n_base_geometries,p['median'],yerr=[p['median']-p['min'],p['max']-p['median']],fmt='o-',color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc])
    ax.set(xlabel='Geometries per replicate',ylabel='Nondominated designs');clean(ax);legend_above(ax,2)
    ax=axs.flat[5];panel(ax,5)
    stress_order=['intrinsic_adiabatic','bounded_moderate','bounded_severe']
    data=[m.loc[m.stress_id==x,'resolved_high_grade_discharge_fraction'] for x in stress_order]
    bp=ax.boxplot(data,positions=np.arange(3),showfliers=False,patch_artist=True)
    for box,c in zip(bp['boxes'],['#3978A8','#E6862A','#D75452']):box.set(facecolor=c,alpha=.58,edgecolor=c)
    ax.set_xticks(np.arange(3),['Adiabatic','Moderate','Severe']);ax.set_ylabel('Deliverable-energy fraction');clean(ax)
    return finish(fig,SUPP/'Supplementary_Figure_S07_robustness_seed_detail',6,['sampling_robustness_v16','bounded_robustness_v16','model_form_stress_v16'])


def supplementary_8(t: dict[str, object]) -> dict[str, object]:
    n=t['nanoparticle_dose_response'];sk=t['skeleton_within_source_effects'];a=t['architecture_gate'];e=t['evidence_coverage'];pair=t['pair_gate'];fig, axs = composite_grid()
    ax=axs.flat[0];panel(ax,0);q=n[n.cp_liquid_J_kgK.notna()]
    source_labels={
        'ANDREU2014':'Andreu-Cabedo et al. (2014)',
        'CHIERUZZI2013':'Chieruzzi et al. (2013)',
        'LU2013':'Lu and Huang (2013)',
        'XIE2016':'Xie et al. (2016)',
    }
    for g,p in q.groupby('study_group'):
        ax.plot(p.loading_wt_pct,p.cp_liquid_J_kgK,'o-',label=source_labels.get(g,g))
    ax.set(xlabel='Loading (wt%)',ylabel='$c_p$ (J kg$^{-1}$ K$^{-1}$)');clean(ax);legend_above(ax,2)
    ax=axs.flat[1];panel(ax,1);q=n[n.nanoparticle!='none'];ax.scatter(q.loading_wt_pct,q.reported_cp_change_pct,c=pd.Categorical(q.nanoparticle).codes,cmap='tab10',s=35);ax.axhline(0,color='#444',lw=.7);ax.set(xlabel='Loading (wt%)',ylabel='Reported $c_p$ change (%)');clean(ax)
    ax=axs.flat[2];panel(ax,2);y=np.arange(len(sk));ax.barh(y,sk.capacity_change_pct_vs_matched_baseline,color='#3978A8',label='Capacity');ax.barh(y,sk.rate_or_transport_change_pct_vs_matched_baseline,color='#E6862A',alpha=.65,label='Rate / transport');ax.set_yticks(y,[candidate_short(x) for x in sk.candidate_id]);ax.set_xlabel('Within-source change (%)');clean(ax);legend_above(ax,2)
    ax=axs.flat[3];panel(ax,3);m=e[['composition_resolved','temperature_window_resolved','capacity_resolved','solid_property_vector_resolved','hydraulic_closure','matched_transient_unit']].sum(axis=1);ax.barh(np.arange(len(e)),m,color='#3E9B65');ax.set_yticks(np.arange(len(e)),[candidate_short(x) for x in e.candidate_id]);ax.set_xlabel('Available transient-model inputs (of 6)');clean(ax)
    ax=axs.flat[4];panel(ax,4);ax.scatter(a.missing_evidence_count,a.within_source_quantitation.astype(int),c=a.dynamic_common_rank_eligible.astype(int),cmap=ListedColormap(['#BFC5CB','#3E9B65']),s=70);ax.set(xlabel='Unavailable model inputs',ylabel='Within-study quantification');ax.set_yticks([0,1],['No','Yes']);clean(ax)
    ax=axs.flat[5];panel(ax,5)
    attrs=['chemical_compatibility_status','wetting_status','cycle_stability_status']
    direct=[];partial=[];open_count=[]
    for col in attrs:
        vals=pair[col].astype(str).str.lower()
        direct.append(int(vals.str.contains('direct|resolved_for').sum()))
        partial.append(int(vals.str.contains('source_specific|partial').sum()))
        open_count.append(int((vals=='unresolved').sum()))
    x=np.arange(3)
    ax.bar(x,direct,color='#4FAF9F',label='Direct')
    ax.bar(x,partial,bottom=direct,color='#F2C35D',label='Partial')
    ax.bar(x,open_count,bottom=np.array(direct)+np.array(partial),color='#D9DDE1',label='Unreported')
    ax.set_xticks(x,['Chemistry','Wetting','Cycling']);ax.set_ylabel('Salt-skeleton pairs');clean(ax);legend_above(ax,3)
    return finish(fig,SUPP/'Supplementary_Figure_S08_architecture_source_detail',6,['nanoparticle_dose_response','skeleton_within_source_effects','architecture_gate','evidence_coverage','salt_skeleton_pair_registry'])


def supplementary_9(t: dict[str, object]) -> dict[str, object]:
    threshold=t['outlet_grade_threshold_sensitivity_v16'];scale=t['module_scale_sensitivity_v16'];transport=t['transport_temperature_sensitivity_v16'];fig, axs = five_panel_grid()

    ax=axs[0];panel(ax,0)
    thresholds=sorted(threshold.threshold_sensitivity_grade_fraction.unique())
    for j,sc in enumerate(SCENARIO_COLOR):
        data=[threshold.loc[(threshold.scenario_id==sc)&np.isclose(threshold.threshold_sensitivity_grade_fraction,g),'parallel_unit_count_ratio_vs_g090'] for g in thresholds]
        pos=np.arange(len(thresholds))+(j-.5)*.18
        bp=ax.boxplot(data,positions=pos,widths=.28,showfliers=False,patch_artist=True,manage_ticks=False)
        for box in bp['boxes']:box.set(facecolor=SCENARIO_COLOR[sc],edgecolor=SCENARIO_COLOR[sc],alpha=.48)
        ax.plot([],[],color=SCENARIO_COLOR[sc],lw=6,alpha=.48,label=SCENARIO_LABEL[sc])
    ax.axhline(1,color='#444444',lw=.7,ls=':');ax.set_xticks(np.arange(len(thresholds)),[f'{g:.2f}' for g in thresholds]);ax.set(xlabel='Normalized outlet-temperature criterion',ylabel='Module-count ratio to criterion 0.90');clean(ax);legend_above(ax,2)

    ax=axs[1];panel(ax,1)
    for sc in SCENARIO_COLOR:
        q=threshold[threshold.scenario_id==sc].groupby('threshold_sensitivity_grade_fraction').installed_volume_ratio_vs_g090.quantile([.05,.5,.95]).unstack()
        x=q.index.to_numpy(float);ax.fill_between(x,q[.05],q[.95],color=SCENARIO_COLOR[sc],alpha=.16);ax.plot(x,q[.5],'o-',color=SCENARIO_COLOR[sc],label=SCENARIO_LABEL[sc])
    ax.axhline(1,color='#444444',lw=.7,ls=':');ax.set(xlabel='Normalized outlet-temperature criterion',ylabel='Installed-volume ratio to criterion 0.90');clean(ax);legend_above(ax,2)

    ax=axs[2];panel(ax,2)
    for (sc,design),q in scale.groupby(['scenario_id','reference_design_id']):
        q=q.sort_values('linear_scale_factor');ax.plot(q.linear_scale_factor,q.parallel_unit_count,'o-',color=SCENARIO_COLOR[sc],alpha=.60,label=SCENARIO_LABEL[sc] if design==scale.loc[scale.scenario_id==sc,'reference_design_id'].iloc[0] else None)
    ax.set_yscale('log');ax.set_xticks(sorted(scale.linear_scale_factor.unique()));ax.set(xlabel='Linear module-scale factor',ylabel='Required module count');clean(ax);legend_above(ax,2)

    ax=axs[3];panel(ax,3)
    q=scale.groupby('linear_scale_factor').agg(bed_med=('unit_bed_pressure_drop_kPa','median'),bed_lo=('unit_bed_pressure_drop_kPa',lambda x:x.quantile(.05)),bed_hi=('unit_bed_pressure_drop_kPa',lambda x:x.quantile(.95)),header_med=('header_pressure_drop_kPa','median')).reset_index()
    ax.fill_between(q.linear_scale_factor,q.bed_lo,q.bed_hi,color='#3978A8',alpha=.18);ax.plot(q.linear_scale_factor,q.bed_med,'o-',color='#3978A8',label='Bed median (5–95%)');ax.plot(q.linear_scale_factor,q.header_med,'s--',color='#E6862A',label='Header median');ax.set_yscale('log');ax.set_xticks(q.linear_scale_factor);ax.set(xlabel='Linear module-scale factor',ylabel='Pressure drop (kPa)');clean(ax);legend_above(ax,2)

    ax=axs[4];panel(ax,4)
    cases=['cold_endpoint','mean','hot_endpoint'];labels=['Cold','Mean','Hot'];x=np.arange(3)
    for metric,color,marker,label in [('installed_volume_change_pct','#3978A8','o','Installed volume'),('bed_pressure_drop_change_pct','#E6862A','s','Bed pressure drop')]:
        med=transport.groupby('transport_temperature_case')[metric].median().reindex(cases);lo=transport.groupby('transport_temperature_case')[metric].quantile(.05).reindex(cases);hi=transport.groupby('transport_temperature_case')[metric].quantile(.95).reindex(cases);ax.errorbar(x,med,yerr=[med-lo,hi-med],fmt=marker+'-',color=color,label=label,capsize=2)
    ax.axhline(0,color='#444444',lw=.7,ls=':');ax.set_xticks(x,labels);ax.set_ylabel('Change from 525 °C properties (%)');clean(ax);legend_above(ax,2)

    return finish(fig,SUPP/'Supplementary_Figure_S09_scale_threshold_and_monotonicity',5,['outlet_grade_threshold_sensitivity_v16','module_scale_sensitivity_v16','transport_temperature_sensitivity_v16'])


CAPTIONS = {
    "Figure 1": "Model validation and control-volume checks before design comparison. (a) German Aerospace Center outlet-temperature benchmark with cooling reserved for independent comparison. (b) Heating, cooling, and full-trace errors with radial-envelope coverage. (c) Axial-grid refinement. (d) Particle-model comparison. (e) Reference control-volume comparison. (f) Differences between 64-cell and 2,048-cell calculations for all 3,840 single-module designs. The 52 reported modular-system designs are independently recalculated with 4,096 cells.",
    "Figure 2": "Matched material contrasts are attenuated from fluid to system scale. (a) Sensible-capacity density and viscosity at the midpoint of the study temperature range; marker area denotes thermal conductivity. (b) Filled-module capacity density. (c) Relation between skeleton porosity and the filled-module/fluid contrast ratio. (d) Median signed logarithmic contrasts and interquartile ranges. (e) System/fluid contrast ratios for both power–energy requirements. (f) All 15,360 pairwise comparisons, including 218 nonzero fluid-level contrasts that produce the same minimum required module count and no reversal; the dashed line denotes equal contrast.",
    "Figure 3": "Structure creates a conditional thermal-hydraulic tradeoff. (a,b) Matched deliverable-energy-fraction changes for fine-hot and coarse-hot gradients. (c) Absolute fine-hot bed pressure-drop change. (d) Matched deliverable-energy and absolute hydraulic responses. (e) Particle regime and front thickness. (f) Two-factor incremental explained variance.",
    "Figure 4": "Module requirements under prescribed power–energy constraints after recalculating each module at its share of total flow. (a) Ratios of required to nominal module power. (b) Deliverable-energy fractions at nominal and module-specific flows. (c) Minimum required module counts under energy and nominal power-rating constraints. (d,e) Nondominated alternatives retained after applying the predefined objective-resolution thresholds for 10 MW, 80 MWh and 50 MW, 300 MWh. (f) Pressure drop along the simplified header route versus required module count; fittings, valves, heat exchangers, and external piping are excluded.",
    "Figure 5": "Skeleton and nanoparticle results are reported on their original experimental bases. (a) Within-study capacity-metric and rate or transport effects for compressed expanded graphite, silicon-carbide foam, and metal foams. (b,c) Nanoparticle loading response and maxima. (d) Source measurement ranges relative to the common study range. (e) Missing transient-model inputs. (f) Chemistry, wetting and cycling evidence for each salt-skeleton pair. Source identifiers and direct or calculated quantities are reported in Supporting Information Section S4.",
    "Figure 6": "Sensitivity tests address distinct sources of uncertainty. (a) Sensitivity to the predefined objective-resolution thresholds used to summarize the nondominated set. (b,c) Independent-sample IGD+ and additive epsilon indicators. (d) Deterministic propagation of source-reported property ranges. (e) Effects of model formulation on volume, deliverable-energy fraction and charge-discharge asymmetry; the severe case does not resolve the 90% response times. (f) Geometry-dependent standby heat loss; the adiabatic reference is omitted because it is identically zero.",
    "Supplementary Figure S1": "Documented fluid-property correlations across the common temperature range. (a–d) Density, heat capacity, viscosity and thermal conductivity. (e) Reported correlation-validity intervals. (f) Normalized capacity and transport descriptors.",
    "Supplementary Figure S2": "Numerical verification. (a) Grid convergence of front thickness. (b) Grid convergence of charge-energy balance. (c) Numerical diffusion under first-order upwind and second-order minmod reconstruction. (d) Independent recalculation of all 52 reported modular-system designs with 4,096 instead of 2,048 cells, normalized by the predefined 0.50% deliverable-energy limit.",
    "Supplementary Figure S3": "Coverage of the defined single-module design space. (a–c) Unique base geometries across vessel, packing, and thermal-input variables; each base geometry is counted once. (d) Fine and coarse layer diameters derived from each base particle size. (e) Marginal coverage of the five sampled variables. (f) Pairwise correlations of the sampled variables.",
    "Supplementary Figure S4": "Single-module response distributions. (a) Salt and solid inventory. (b) Deliverable-energy fraction. (c) Pressure drop. (d) Charge time. (e) Deliverable-energy fraction versus frictional hydraulic-energy fraction. (f) Volumetric capacity versus salt inventory.",
    "Supplementary Figure S5": "Fluid-resolved particle-grading response without repeated one-fluid plots. (a,b) Fluid-envelope and median deliverable-energy response across hot-side fractions for fine-hot and coarse-hot grading. (c) Fluid-by-orientation median deliverable-energy changes. (d) Matched median bed pressure-drop and deliverable-energy changes for both orientations.",
    "Supplementary Figure S6": "Objective projections and required module count. (a,b) Packed-volume–hydraulic-energy and salt-inventory–excess-capacity projections for the 10 MW, 80 MWh requirement. (c,d) Corresponding projections for the 50 MW, 300 MWh requirement. (e) Fluid and particle-arrangement identities across predefined objective-resolution thresholds. (f) Required module count.",
    "Supplementary Figure S7": "Sampling and physical-model sensitivity. (a–d) Independent-replicate IGD+ and additive epsilon indicators for both power–energy requirements. (e) Nondominated designs identified in each independent sample. (f) Deliverable-energy fraction under intrinsic, moderate, and severe physical-model assumptions.",
    "Supplementary Figure S8": "Material evidence detail. (a) Nanoparticle heat capacity by source. (b) Reported nanoparticle heat-capacity changes. (c) Skeleton capacity-metric and rate or transport effects within source. (d) Available transient-model inputs. (e) Availability of within-study and common-model evidence. (f) Direct, partial, and unreported salt–skeleton evidence. Source locator and direct or calculated status are reported in Section S4.",
    "Supplementary Figure S9": "Outlet-temperature, module-size and temperature-dependent-property sensitivity. (a,b) Module-count and installed-volume sensitivity to outlet-temperature criteria. (c) Required module count under geometric scaling of the 52 reported designs. (d) Bed and simplified-header pressure drops. (e) Sensitivity to transport properties evaluated at the cold, mean, and hot temperatures.",
}


def contact_sheet(paths: list[Path], out: Path, columns: int = 2) -> None:
    thumbs=[]
    for p in paths:
        im=Image.open(p).convert('RGB');im.thumbnail((1500,900));thumbs.append((p.stem,im.copy()))
    rows=math.ceil(len(thumbs)/columns);cell_w=1520;cell_h=960
    sheet=Image.new('RGB',(columns*cell_w,rows*cell_h),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(thumbs):
        x=(i%columns)*cell_w;y=(i//columns)*cell_h;draw.text((x+12,y+8),name,fill='black');sheet.paste(im,(x+10,y+45))
    sheet.save(out,dpi=(150,150))


def publication_captions() -> str:
    """Read the quantitative captions used by the manuscript and SI."""
    manuscript = ROOT / "manuscript_package" / "submission_files" / "Manuscript_Source_V18.md"
    supporting = ROOT / "manuscript_package" / "submission_files" / "Supporting_Information_Source_V18.md"
    if not manuscript.exists() or not supporting.exists():
        return '\n\n'.join(f"{k}. {v}" for k, v in CAPTIONS.items()) + '\n'
    main_lines = [
        line.replace("**", "").strip()
        for line in manuscript.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\*\*Figure [1-6]\.", line)
    ]
    supp_lines = [
        line.strip()
        for line in supporting.read_text(encoding="utf-8").splitlines()
        if re.match(r"^Supplementary Figure S[1-9]\.", line)
    ]
    if len(main_lines) != 6 or len(supp_lines) != 9:
        raise RuntimeError(f"Expected 6 main and 9 supplementary captions, got {len(main_lines)} and {len(supp_lines)}")
    return '\n\n'.join(main_lines + supp_lines) + '\n'


def main() -> None:
    # Remove only generated figure/table artefacts from the same V16 package.
    for folder in (MAIN, SUPP, CONTACT, TABLES):
        for p in folder.iterdir():
            if p.is_file(): p.unlink()
    t=load();copy_release_tables()
    manifest=[]
    for builder in (figure_1,figure_2,figure_3,figure_4,figure_5,figure_6,
                    supplementary_1,supplementary_2,supplementary_3,supplementary_4,
                    supplementary_5,supplementary_6,supplementary_7,supplementary_8,
                    supplementary_9):
        manifest.append(builder(t))
    (OUT/'figures'/'figure_manifest_v16.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (OUT/'FIGURE_CAPTIONS_V16.md').write_text(publication_captions(),encoding='utf-8')
    source_rows=[]
    for rec in manifest:
        for source in rec['sources']:source_rows.append({'figure_id':rec['figure_id'],'source_table_or_module':source,'panel_count':rec['panel_count']})
    pd.DataFrame(source_rows).to_csv(TABLES/'figure_source_map_v16.csv',index=False)
    print(json.dumps({'status':'PASS','main_figures':6,'supplementary_figures':9,
                      'formats':['pdf'],
                      'manifest':str(OUT/'figures'/'figure_manifest_v16.json')},indent=2))


if __name__=='__main__':
    main()
