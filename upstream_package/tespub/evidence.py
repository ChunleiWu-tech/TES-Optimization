"""Build evidence-gated material, skeleton and architecture comparison tables.

The module never converts a literature percentage into a universal model
multiplier. Cross-study rows are used for coverage and mechanism mapping only;
numerical effects remain normalised to the matched baseline in the same source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(r"^\s*$", np.nan, regex=True), errors="coerce")


def _effect_row(
    architecture: str,
    candidate: str,
    source: str,
    capacity_change_pct: float | None,
    rate_or_transport_change_pct: float | None,
    transport_metric: str,
    temperature_window: str,
    evidence_scope: str,
) -> dict[str, Any]:
    return {
        "architecture_family": architecture,
        "candidate_id": candidate,
        "source_id": source,
        "capacity_change_pct_vs_matched_baseline": capacity_change_pct,
        "rate_or_transport_change_pct_vs_matched_baseline": rate_or_transport_change_pct,
        "rate_or_transport_metric": transport_metric,
        "temperature_window_C": temperature_window,
        "comparison_scope": evidence_scope,
        "cross_architecture_rank_eligible": False,
    }


def build_within_source_skeleton_effects(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    outcomes = pd.read_csv(data_dir / "skeleton_reported_outcomes.csv")
    numeric = pd.to_numeric(outcomes["value"], errors="coerce")
    values = dict(zip(outcomes.loc[numeric.notna(), "outcome_id"], numeric[numeric.notna()].astype(float)))
    ceg_capacity = 100.0 * (values["EG_LATENT_DENSITY"] / values["NEAT_LATENT_DENSITY"] - 1.0)
    neat_eff = values["NEAT_EFFUSIVITY"]
    rows = [
        _effect_row(
            "infiltrated_CEG_matrix", "CEG_SOLAR_SALT_60CR", "KUMAR_CEG_SOLAR_SALT_2024",
            ceg_capacity, 250.0, "thermal conductivity midpoint of reported 3-4 fold interval",
            "50-300; conductivity measured near ambient", "same-source neat Solar Salt baseline",
        ),
        _effect_row(
            "open_cell_SiC_foam", "SIC_FOAM_SOLAR_SALT", "ZHANG_SIC_SHELL_TUBE_2023",
            values["SIC_UNIT_STORED_ENERGY"], values["SIC_UNIT_STORAGE_RATE"],
            "unit energy-storage rate", "phase change near 220", "same-source no-foam unit",
        ),
        _effect_row(
            "open_cell_Ni_foam", "NI_FOAM_SOLAR_SALT_G1", "XIAO_METAL_FOAM_SOLAR_SALT_2020",
            100.0 * (1583.0 / 1677.0 - 1.0), 100.0 * (values["NI_EFFUSIVITY"] / neat_eff - 1.0),
            "measured liquid thermal effusivity", "260-290", "same-source neat Solar Salt baseline",
        ),
        _effect_row(
            "open_cell_Cu_foam", "CU_FOAM_SOLAR_SALT_G1", "XIAO_METAL_FOAM_SOLAR_SALT_2020",
            100.0 * (1371.0 / 1677.0 - 1.0), 100.0 * (values["CU_EFFUSIVITY"] / neat_eff - 1.0),
            "measured liquid thermal effusivity", "260-290", "same-source neat Solar Salt baseline",
        ),
        {
            **_effect_row(
                "ceramic_sphere_packed_bed", "CERAMIC_SPHERE_PACKED_BED", "WEISS_PACKED_BED_SOLAR_SALT_2024",
                0.0, 0.0, "reference for the executable dynamic branch", "290-390 experiment; 500-550 compatibility context",
                "source-complete reference geometry and filler properties",
            ),
            "cross_architecture_rank_eligible": True,
        },
    ]
    return pd.DataFrame(rows)


def build_nanoparticle_dose_response(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    registry = pd.read_csv(data_dir / "nanoparticle_material_registry.csv")
    registry["loading_wt_pct"] = _numeric(registry["loading_wt_pct"])
    registry["cp_liquid_J_kgK"] = _numeric(registry["cp_liquid_J_kgK"])
    registry["cp_sd_J_kgK"] = _numeric(registry["cp_sd_J_kgK"])
    registry["reported_cp_change_pct"] = _numeric(registry["reported_cp_change_pct"])
    selected = registry[
        registry["study_group"].isin(["ANDREU2014", "XIE2016", "CHIERUZZI2013", "LU2013"])
        & registry["loading_wt_pct"].notna()
    ].copy()
    selected["temperature_C"] = np.nan
    selected["nanoparticle"] = selected["nanoparticle_or_structure"]
    selected["comparison_mode"] = "within_source"
    selected["dynamic_transport_complete"] = selected["transport_complete_for_dynamic"].astype(bool)
    selected = selected.rename(columns={"temperature_window_C": "temperature_window_C_source"})
    base_cols = [
        "study_group", "source_id", "state_id", "base_salt", "nanoparticle", "loading_wt_pct",
        "temperature_C", "temperature_window_C_source", "cp_liquid_J_kgK", "cp_sd_J_kgK",
        "reported_cp_change_pct", "evidence_mode", "comparison_mode", "dynamic_transport_complete",
    ]

    responses = pd.read_csv(data_dir / "nanoparticle_reported_responses.csv")
    responses["temperature_C"] = _numeric(responses["temperature_C"])
    responses["reported_relative_change_pct"] = _numeric(responses["reported_relative_change_pct"])
    mask = (
        responses["source_id"].eq("LASFARGUES_CUO_TIO2_NITRATE_2015")
        & responses["temperature_C"].eq(440.0)
        & responses["property_name"].eq("specific_heat_relative_change")
    )
    extra_rows: list[dict[str, Any]] = []
    for row in responses.loc[mask].itertuples():
        match = re.search(r"_(\d+)P(\d+)$", str(row.state_id))
        if not match:
            continue
        loading = float(f"{match.group(1)}.{match.group(2)}")
        particle = "CuO" if "CUO" in str(row.state_id) else "TiO2"
        extra_rows.append({
            "study_group": f"LASFARGUES2015_{particle}",
            "source_id": row.source_id,
            "state_id": row.state_id,
            "base_salt": "Solar Salt",
            "nanoparticle": particle,
            "loading_wt_pct": loading,
            "temperature_C": 440.0,
            "temperature_window_C_source": "265-440",
            "cp_liquid_J_kgK": np.nan,
            "cp_sd_J_kgK": np.nan,
            "reported_cp_change_pct": float(row.reported_relative_change_pct),
            "evidence_mode": "exact_reported_table_relative_change",
            "comparison_mode": "within_source",
            "dynamic_transport_complete": False,
        })
    extra = pd.DataFrame(extra_rows, columns=base_cols)
    return pd.concat([selected[base_cols], extra], ignore_index=True).sort_values(
        ["study_group", "nanoparticle", "loading_wt_pct"], kind="stable"
    )


def build_architecture_gate(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    registry = json.loads((data_dir / "architecture_registry.json").read_text(encoding="utf-8"))
    rows = []
    for item in registry["architectures"]:
        missing = item.get("required_missing_evidence", [])
        status = str(item["status"])
        rows.append({
            "architecture_id": item["id"],
            "implementation_status": status,
            "dynamic_common_rank_eligible": status == "implemented_runnable",
            "within_source_quantitation": item["id"] in {"sensible_packed_bed_LTNE", "open_cell_foam_pcm"},
            "missing_evidence_count": len(missing),
            "missing_evidence": " | ".join(missing),
            "governing_physics": " | ".join(item.get("governing_physics", [])),
            "benchmark_source": " | ".join(item.get("benchmark_sources", [item.get("benchmark_source", item.get("validation_source", ""))])),
        })
    return pd.DataFrame(rows)


def build_evidence_coverage(data_dir: str | Path) -> pd.DataFrame:
    skeleton = pd.read_csv(Path(data_dir) / "skeleton_material_evidence.csv")
    coverage = []
    for row in skeleton.itertuples():
        complete_solid = all(pd.notna(v) for v in [row.solid_density_kg_m3, row.solid_cp_J_kgK, row.solid_thermal_conductivity_W_mK])
        coverage.append({
            "candidate_id": row.candidate_id,
            "architecture_family": row.architecture_family,
            "composition_resolved": 1,
            "temperature_window_resolved": int(bool(str(row.temperature_window_C).strip())),
            "capacity_resolved": int(pd.notna(row.composite_cp_liquid_J_kgK) or pd.notna(row.latent_heat_kJ_kg) or complete_solid),
            "solid_property_vector_resolved": int(complete_solid),
            "hydraulic_closure": int(row.candidate_id == "CERAMIC_SPHERE_PACKED_BED"),
            "matched_transient_unit": int(row.candidate_id in {"CERAMIC_SPHERE_PACKED_BED", "SIC_FOAM_SOLAR_SALT"}),
            "dynamic_common_rank_eligible": int(row.candidate_id == "CERAMIC_SPHERE_PACKED_BED"),
            "evidence_tier": row.evidence_tier,
            "source_id": row.source_id,
        })
    return pd.DataFrame(coverage)


def build_evidence_outputs(data_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "skeleton_within_source_effects": build_within_source_skeleton_effects(data_dir),
        "nanoparticle_dose_response": build_nanoparticle_dose_response(data_dir),
        "architecture_gate": build_architecture_gate(data_dir),
        "evidence_coverage": build_evidence_coverage(data_dir),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    summary = {
        "status": "PASS_WITH_EXPLICIT_DATA_GATES",
        "n_skeleton_candidates": int(len(tables["evidence_coverage"])),
        "n_architectures": int(len(tables["architecture_gate"])),
        "n_common_dynamic_architectures": int(tables["architecture_gate"]["dynamic_common_rank_eligible"].sum()),
        "n_nanoparticle_states": int(len(tables["nanoparticle_dose_response"])),
        "cross_architecture_rule": "No direct global ranking across skeleton architectures. All effect sizes are normalised to matched within-source baselines; only the source-complete ceramic packed-bed branch enters the common dynamic Pareto analysis.",
    }
    (out_dir / "evidence_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
