"""Continuous service closure and integer design-consequence audit for V18.

The V16 physical model and its integer service-sizing results are treated as a
frozen baseline.  This module introduces only one analysis coordinate: the
module count may be real-valued while locating the continuous energy closure.
Every function evaluation reuses the registered V16 LTNE transient kernel at
the corresponding per-module flow.  Physical designs remain integer-valued.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from .codesign_v16 import _registered_ambient_temperature_K, load_config
from .fluid_properties import fluid_properties
from .ltne_v16 import BedGeometryV16, evaluate_packed_bed_v16


CONTINUOUS_FILENAME = "01_continuous_service_closure.csv"
PAIRWISE_FILENAME = "02_pairwise_design_consequence.csv"
SUMMARY_FILENAME = "03_design_consequence_summary.csv"
QC_JSON_FILENAME = "05_V18_QC_report.json"
QC_TEXT_FILENAME = "05_V18_QC_report.txt"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_with_tolerance(value: float, relative_tolerance: float) -> int:
    """Apply the registered integer convention without floating-point jitter."""
    scale = max(1.0, abs(float(value)))
    nearest = round(float(value))
    adjusted = float(nearest) if abs(float(value) - nearest) <= relative_tolerance * scale else float(value)
    return int(math.ceil(adjusted))


def _scenario_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["scenario_id"]): dict(row) for row in config["service_scenarios"]}


def _case_identifier(scenario_id: str, design_id: str) -> str:
    return f"{scenario_id}::{design_id}"


def _source_geometry(source: dict[str, Any]) -> BedGeometryV16:
    return BedGeometryV16(
        internal_diameter_m=float(source["internal_diameter_m"]),
        bed_height_m=float(source["bed_height_m"]),
        porosity=float(source["porosity"]),
        particle_diameter_m=float(source["particle_diameter_m"]),
        topology=str(source["topology"]),
        gradient_ratio=float(source["gradient_ratio"]),
        gradient_orientation=str(source["gradient_orientation"]),
        layer_fraction=float(source["layer_fraction"]),
    )


def _solve_continuous_case(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    source, service, scenario, config = task
    boundary = config["design_consequence_boundary"]
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    delta_t = hot_K - cold_K
    power_MW = float(scenario["target_thermal_power_MW"])
    duration_h = float(scenario["storage_duration_h"])
    required_energy = power_MW * duration_h
    transport_temperature = scenario.get("transport_property_temperature_K")
    multipliers = {
        "density": float(source.get("density_multiplier", 1.0)),
        "cp": float(source.get("cp_multiplier", 1.0)),
        "viscosity": float(source.get("viscosity_multiplier", 1.0)),
        "conductivity": float(source.get("conductivity_multiplier", 1.0)),
    }
    if transport_temperature is None:
        cp = (
            float(source["mean_cp_J_kgK"])
            if "mean_cp_J_kgK" in source and pd.notna(source["mean_cp_J_kgK"])
            else fluid_properties(str(source["fluid_model_id"]), 0.5 * (cold_K + hot_K)).cp_J_kgK
            * multipliers["cp"]
        )
    else:
        cp = (
            fluid_properties(str(source["fluid_model_id"]), float(transport_temperature)).cp_J_kgK
            * multipliers["cp"]
        )
    total_flow = power_MW * 1e6 / (cp * delta_t)
    unit_power_MW = float(source["nominal_thermal_input_kW"]) / 1000.0
    n_power_cont = power_MW / max(unit_power_MW, 1e-12)
    n_star = int(service["parallel_unit_count"])
    geometry = _source_geometry(source)
    solid = config["reference_geometry"]
    cache: dict[float, tuple[float, float]] = {}

    def evaluate_count(count: float) -> tuple[float, float]:
        count = float(count)
        key = round(count, 12)
        if key not in cache:
            actual = evaluate_packed_bed_v16(
                model_id=str(source["fluid_model_id"]),
                design_id=(
                    f"{source['design_id']}__{scenario['scenario_id']}__V18_N{count:.10f}"
                ),
                geometry=geometry,
                mass_flow_kg_s=total_flow / count,
                cold_temperature_K=cold_K,
                hot_temperature_K=hot_K,
                solid_density_kg_m3=float(solid["solid_density_kg_m3"]),
                solid_cp_J_kgK=float(solid["solid_cp_J_kgK"]),
                solid_thermal_conductivity_W_mK=float(solid["solid_thermal_conductivity_W_mK"]),
                n_cells=int(source["n_cells"]),
                axial_dispersion_m2_s=float(source.get("axial_dispersion_m2_s", 0.0)),
                solid_axial_diffusivity_m2_s=float(source.get("solid_axial_diffusivity_m2_s", 0.0)),
                wall_loss_W_m3K=float(source.get("wall_loss_W_m3K", 0.0)),
                ambient_temperature_K=_registered_ambient_temperature_K(source, config),
                minimum_outlet_grade_fraction=float(scenario["minimum_outlet_grade_fraction"]),
                property_multipliers=multipliers,
                transport_property_temperature_K=(
                    None if transport_temperature is None else float(transport_temperature)
                ),
            )
            unit_energy = float(actual["stored_heat_MWh"]) * float(actual["high_grade_discharge_fraction"])
            cache[key] = (count * unit_energy - required_energy, unit_energy)
        return cache[key]

    # The integer result is the validated feasible upper bracket.  Reuse its
    # stored capacity exactly so this added analysis cannot alter the baseline.
    upper = float(n_star)
    upper_unit_energy = float(service["installed_high_grade_capacity_MWh"]) / upper
    upper_g = float(service["installed_high_grade_capacity_MWh"]) - required_energy
    cache[round(upper, 12)] = (upper_g, upper_unit_energy)
    relative_energy_tolerance = float(boundary["root_relative_energy_tolerance"])
    if upper_g < -relative_energy_tolerance * required_energy:
        raise RuntimeError(
            f"validated integer count is energy-infeasible for {_case_identifier(str(scenario['scenario_id']), str(source['design_id']))}"
        )

    minimum_count = float(boundary.get("minimum_continuous_count", 1e-3))
    # The quantity needed for the design-consequence boundary is the
    # continuous crossing adjacent to the validated minimum integer count.
    # A wider bracket can contain earlier numerical crossings when the
    # time-discrete LTNE response has small local oscillations; selecting that
    # earlier crossing would no longer describe the N*-1 -> N* decision.
    lower = max(minimum_count, upper - 1.0)
    if not lower < upper:
        lower = max(minimum_count, 0.5 * upper)
    lower_g, _ = evaluate_count(lower)
    bracket_steps = 1
    while lower_g > 0.0 and lower > minimum_count * (1.0 + 1e-12):
        upper = lower
        upper_g = lower_g
        lower = max(minimum_count, 0.5 * lower)
        lower_g, _ = evaluate_count(lower)
        bracket_steps += 1
    if lower_g > 0.0:
        raise RuntimeError(
            f"energy root lies below registered model count for {_case_identifier(str(scenario['scenario_id']), str(source['design_id']))}"
        )

    # Do not snap a slightly infeasible lower endpoint to the root merely
    # because its energy residual is small.  Such snapping can move the
    # continuous coordinate onto N*-1 and therefore changes its tolerance-
    # aware ceiling, even though the validated integer solution remains N*.
    # The sign bracket, rather than the energy tolerance, defines the root.
    n_energy_cont = float(
        brentq(
            lambda value: evaluate_count(float(value))[0],
            lower,
            upper,
            xtol=float(boundary["root_count_absolute_tolerance"]),
            rtol=float(boundary["root_count_relative_tolerance"]),
            maxiter=160,
        )
    )
    root_g, root_unit_energy = evaluate_count(n_energy_cont)
    if abs(root_g) > relative_energy_tolerance * required_energy:
        # The transient event detector can make G(N) weakly nonsmooth at the
        # 1e-5 relative-energy scale.  Polish only such exceptional roots by
        # minimizing the registered closure residual inside the same sign
        # bracket; never relax the preregistered acceptance threshold.
        polished = minimize_scalar(
            lambda value: abs(evaluate_count(float(value))[0]),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": float(boundary["root_count_absolute_tolerance"]), "maxiter": 120},
        )
        polished_count = float(polished.x)
        polished_g, polished_unit_energy = evaluate_count(polished_count)
        if abs(polished_g) < abs(root_g):
            n_energy_cont = polished_count
            root_g = polished_g
            root_unit_energy = polished_unit_energy
    ceiling_relative_tolerance = float(boundary["integer_ceiling_relative_tolerance"])
    ceiling_band = ceiling_relative_tolerance * max(1.0, abs(lower))
    if lower_g < 0.0 and n_energy_cont - lower <= ceiling_band:
        # Brent may return the infeasible endpoint when the crossing lies
        # inside its floating-point termination band.  Preserve the strict
        # right-sided location implied by G(lower)<0 without claiming more
        # root precision than the registered ceiling convention supports.
        n_energy_cont = min(upper, lower + 2.0 * max(ceiling_band, np.finfo(float).eps * abs(lower)))
        root_g, root_unit_energy = evaluate_count(n_energy_cont)
    n_cont = max(1.0, n_energy_cont, n_power_cont)
    ceil_tol_n_cont = ceil_with_tolerance(n_cont, ceiling_relative_tolerance)
    constraint_tolerance = float(boundary["root_count_absolute_tolerance"])
    if abs(n_energy_cont - n_power_cont) <= constraint_tolerance:
        governing = "co_limited"
    elif n_energy_cont > n_power_cont:
        governing = "energy_constrained"
    else:
        governing = "power_rating_constrained"
    old_governing = (
        "power_rating_constrained"
        if int(service["power_limited_unit_count"]) > int(service["resolved_energy_limited_unit_count"])
        else "energy_constrained"
    )
    minimum_duty = float(config["service_model"].get("minimum_resolved_to_nominal_duty_ratio", 0.05))
    maximum_count = min(
        float(config["service_model"]["maximum_unit_count"]),
        power_MW / max(minimum_duty * unit_power_MW, 1e-12),
    )
    return {
        "case_id": _case_identifier(str(scenario["scenario_id"]), str(source["design_id"])),
        "scenario_id": str(scenario["scenario_id"]),
        "target_thermal_power_MW": power_MW,
        "storage_duration_h": duration_h,
        "required_energy_MWh": required_energy,
        "design_id": str(source["design_id"]),
        "base_design_id": str(source["base_design_id"]),
        "topology_id": str(source["topology_id"]),
        "topology": str(source["topology"]),
        "gradient_orientation": str(source["gradient_orientation"]),
        "fluid_model_id": str(source["fluid_model_id"]),
        "fluid_name": str(source["fluid_name"]),
        "n_cells": int(source["n_cells"]),
        "N_star_old": n_star,
        "N_E_cont": n_energy_cont,
        "N_P_cont": n_power_cont,
        "N_cont": n_cont,
        "governing_constraint": governing,
        "old_governing_constraint": old_governing,
        "ceil_tol_Ncont": ceil_tol_n_cont,
        "N_star_match": bool(ceil_tol_n_cont == n_star),
        "root_residual_MWh": root_g,
        "root_relative_residual": abs(root_g) / required_energy,
        "root_unit_deliverable_energy_MWh": root_unit_energy,
        "resolved_unit_flow_at_Ncont_kg_s": total_flow / n_cont,
        "resolved_unit_power_at_Ncont_kW": power_MW * 1000.0 / n_cont,
        "resolved_to_nominal_duty_ratio_at_Ncont": (power_MW / n_cont) / max(unit_power_MW, 1e-12),
        "maximum_continuous_count": maximum_count,
        "within_registered_count_and_duty_limits": bool(n_cont <= maximum_count + constraint_tolerance),
        "root_lower_bracket": lower,
        "root_upper_bracket": upper,
        "root_bracket_steps": bracket_steps,
        "eval_count": len(cache),
        "analysis_only_fractional_count": True,
        "continuous_count_definition": "max(adjacent-boundary self-consistent energy root, continuous nominal-power lower bound, 1)",
    }


def _prepare_tasks(
    unit: pd.DataFrame,
    service: pd.DataFrame,
    config: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    unit_index = unit.set_index("design_id", drop=False)
    scenarios = _scenario_by_id(config)
    tasks = []
    for record in service.to_dict("records"):
        source = unit_index.loc[str(record["design_id"])].to_dict()
        scenario = scenarios[str(record["scenario_id"])]
        tasks.append((source, record, scenario, config))
    return tasks


def stratified_qc_tasks(
    tasks: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    metadata = pd.DataFrame(
        {
            "index": np.arange(len(tasks)),
            "scenario_id": [str(t[1]["scenario_id"]) for t in tasks],
            "fluid_model_id": [str(t[0]["fluid_model_id"]) for t in tasks],
            "old_constraint": [
                "power"
                if int(t[1]["power_limited_unit_count"]) > int(t[1]["resolved_energy_limited_unit_count"])
                else "energy"
                for t in tasks
            ],
            "N_star": [int(t[1]["parallel_unit_count"]) for t in tasks],
        }
    )
    selected: set[int] = set()
    for _, part in metadata.groupby(["scenario_id", "fluid_model_id", "old_constraint"], sort=True):
        ordered = part.sort_values("N_star")
        for position in (0, len(ordered) // 2, len(ordered) - 1):
            if len(ordered):
                selected.add(int(ordered.iloc[position]["index"]))
    return [tasks[index] for index in sorted(selected)]


def run_continuous_service_closure(
    unit: pd.DataFrame,
    service: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    qc_only: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    """Run a checkpointed, deterministic outer-parallel continuous closure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir.parent / ".v18_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partial = checkpoint_dir / (
        "continuous_service_closure_qc.partial.csv" if qc_only else "continuous_service_closure.partial.csv"
    )
    tasks = _prepare_tasks(unit, service, config)
    if qc_only:
        tasks = stratified_qc_tasks(tasks)
    completed: set[str] = set()
    frames: list[pd.DataFrame] = []
    if partial.is_file():
        previous = pd.read_csv(partial)
        if not previous.empty:
            # A completed checkpoint may still contain rows that fail the
            # registered numerical gates after a solver-method upgrade.  Keep
            # every accepted row and re-evaluate only the invalid cases.  This
            # preserves the expensive full run while making repair explicit,
            # deterministic and resumable.
            root_limit = float(
                config["design_consequence_boundary"]["root_relative_energy_tolerance"]
            )
            expected_counts = {
                _case_identifier(str(task[1]["scenario_id"]), str(task[0]["design_id"])):
                int(task[1]["parallel_unit_count"])
                for task in tasks
            }
            checkpoint_count_mismatch = previous.apply(
                lambda row: int(row["N_star_old"])
                != expected_counts.get(str(row["case_id"]), int(row["N_star_old"])),
                axis=1,
            )
            invalid = (
                ~previous["N_star_match"].astype(bool)
                | (previous["root_relative_residual"].astype(float) > root_limit)
                | ~previous["within_registered_count_and_duty_limits"].astype(bool)
                | checkpoint_count_mismatch
            )
            if invalid.any():
                previous = previous.loc[~invalid].copy()
                previous.to_csv(partial, index=False)
            completed = set(previous["case_id"].astype(str))
            frames.append(previous)
    pending = [
        task for task in tasks
        if _case_identifier(str(task[1]["scenario_id"]), str(task[0]["design_id"])) not in completed
    ]
    boundary = config["design_consequence_boundary"]
    worker_count = int(workers or boundary.get("parallel_workers", 1))
    checkpoint_rows = int(boundary.get("checkpoint_rows", 32))
    start = time.perf_counter()
    processed = len(completed)
    for offset in range(0, len(pending), checkpoint_rows):
        batch = pending[offset : offset + checkpoint_rows]
        if worker_count > 1 and len(batch) > 1:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                rows = list(executor.map(_solve_continuous_case, batch, chunksize=1))
        else:
            rows = [_solve_continuous_case(task) for task in batch]
        frame = pd.DataFrame(rows)
        frame.to_csv(partial, mode="a", header=not partial.exists() or partial.stat().st_size == 0, index=False)
        frames.append(frame)
        processed += len(frame)
        elapsed = time.perf_counter() - start
        rate = (processed - len(completed)) / max(elapsed, 1e-9)
        remaining = len(tasks) - processed
        eta_h = remaining / max(rate, 1e-9) / 3600.0
        print(
            f"[V18 continuous] {processed}/{len(tasks)} persisted; elapsed={elapsed/3600.0:.2f} h; eta={eta_h:.2f} h",
            flush=True,
        )
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result = (
            result.drop_duplicates("case_id", keep="last")
            .sort_values(["scenario_id", "design_id"])
            .reset_index(drop=True)
        )
    final_name = "01_continuous_service_closure_qc.csv" if qc_only else CONTINUOUS_FILENAME
    result.to_csv(output_dir / final_name, index=False)
    return result


def build_pairwise_design_consequence(
    unit: pd.DataFrame,
    service: pd.DataFrame,
    continuous: pd.DataFrame,
    baseline_pairwise: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    keys = ["scenario_id", "base_design_id", "topology_id"]
    unit_metrics = unit.copy()
    unit_metrics["dynamic_energy_density_MWh_m3"] = (
        unit_metrics["stored_heat_MWh"]
        * unit_metrics["high_grade_discharge_fraction"]
        / unit_metrics["storage_volume_m3"]
    )
    merged = service.merge(
        continuous[[
            "scenario_id", "design_id", "N_E_cont", "N_P_cont", "N_cont",
            "governing_constraint", "ceil_tol_Ncont", "N_star_match",
        ]],
        on=["scenario_id", "design_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        unit_metrics[[
            "design_id", "fluid_volumetric_sensible_kWh_m3", "volumetric_capacity_MWh_m3",
            "dynamic_energy_density_MWh_m3",
        ]],
        on="design_id",
        how="left",
        validate="many_to_one",
    )
    left = merged.rename(columns={
        "fluid_model_id": "fluid_model_id_i", "fluid_name": "fluid_name_i",
        "N_E_cont": "N_E_cont_i", "N_P_cont": "N_P_cont_i", "N_cont": "N_cont_i",
        "parallel_unit_count": "Nstar_i", "governing_constraint": "governing_constraint_i",
        "fluid_volumetric_sensible_kWh_m3": "fluid_capacity_i_kWh_m3",
        "volumetric_capacity_MWh_m3": "filled_capacity_i_MWh_m3",
        "dynamic_energy_density_MWh_m3": "dynamic_capacity_i_MWh_m3",
        "service_compactness_MWh_m3": "system_service_density_i_MWh_m3",
    })
    right = merged.rename(columns={
        "fluid_model_id": "fluid_model_id_j", "fluid_name": "fluid_name_j",
        "N_E_cont": "N_E_cont_j", "N_P_cont": "N_P_cont_j", "N_cont": "N_cont_j",
        "parallel_unit_count": "Nstar_j", "governing_constraint": "governing_constraint_j",
        "fluid_volumetric_sensible_kWh_m3": "fluid_capacity_j_kWh_m3",
        "volumetric_capacity_MWh_m3": "filled_capacity_j_MWh_m3",
        "dynamic_energy_density_MWh_m3": "dynamic_capacity_j_MWh_m3",
        "service_compactness_MWh_m3": "system_service_density_j_MWh_m3",
    })
    keep_i = keys + [
        "fluid_model_id_i", "fluid_name_i", "N_E_cont_i", "N_P_cont_i", "N_cont_i", "Nstar_i",
        "governing_constraint_i", "fluid_capacity_i_kWh_m3", "filled_capacity_i_MWh_m3",
        "dynamic_capacity_i_MWh_m3", "system_service_density_i_MWh_m3",
    ]
    keep_j = keys + [
        "fluid_model_id_j", "fluid_name_j", "N_E_cont_j", "N_P_cont_j", "N_cont_j", "Nstar_j",
        "governing_constraint_j", "fluid_capacity_j_kWh_m3", "filled_capacity_j_MWh_m3",
        "dynamic_capacity_j_MWh_m3", "system_service_density_j_MWh_m3",
    ]
    pairs = left[keep_i].merge(right[keep_j], on=keys, how="inner")
    pairs = pairs[pairs["fluid_model_id_i"] < pairs["fluid_model_id_j"]].copy()
    pairs["group_id"] = pairs[keys].astype(str).agg("::".join, axis=1)
    pairs["pair_id"] = pairs["group_id"] + "::" + pairs["fluid_model_id_i"] + "::" + pairs["fluid_model_id_j"]
    pairs["N_min_cont"] = pairs[["N_cont_i", "N_cont_j"]].min(axis=1)
    pairs["N_max_cont"] = pairs[["N_cont_i", "N_cont_j"]].max(axis=1)
    pairs["delta_N_cont"] = pairs["N_max_cont"] - pairs["N_min_cont"]
    ceiling_tolerance = float(config["design_consequence_boundary"]["integer_ceiling_relative_tolerance"])
    pairs["next_integer_distance"] = [
        ceil_with_tolerance(value, ceiling_tolerance) - value
        for value in pairs["N_min_cont"].to_numpy(float)
    ]
    pairs["integer_boundary_margin"] = pairs["delta_N_cont"] - pairs["next_integer_distance"]
    pairs["same_integer"] = pairs["Nstar_i"].astype(int).eq(pairs["Nstar_j"].astype(int))
    pairs["delta_Nstar"] = (pairs["Nstar_i"].astype(int) - pairs["Nstar_j"].astype(int)).abs()
    sign_tolerance = float(config["design_consequence_boundary"]["boundary_sign_tolerance"])
    pairs["boundary_predicts_same_integer"] = pairs["integer_boundary_margin"] <= sign_tolerance
    pairs["boundary_identity_match"] = pairs["boundary_predicts_same_integer"].eq(pairs["same_integer"])
    for scale, i_col, j_col in (
        ("fluid", "fluid_capacity_i_kWh_m3", "fluid_capacity_j_kWh_m3"),
        ("filled", "filled_capacity_i_MWh_m3", "filled_capacity_j_MWh_m3"),
        ("dynamic", "dynamic_capacity_i_MWh_m3", "dynamic_capacity_j_MWh_m3"),
        ("system", "system_service_density_i_MWh_m3", "system_service_density_j_MWh_m3"),
    ):
        pairs[f"A_{scale}_log"] = np.log(pairs[i_col] / pairs[j_col])
    pairs["R_system_fluid"] = np.where(
        pairs["A_fluid_log"].abs() > 1e-14,
        pairs["A_system_log"] / pairs["A_fluid_log"],
        np.nan,
    )
    pairs["N_mid_cont"] = 0.5 * (pairs["N_cont_i"] + pairs["N_cont_j"])
    pairs["one_module_relative_resolution"] = 1.0 / pairs["N_mid_cont"]
    baseline = baseline_pairwise[[
        *keys, "fluid_model_id_i", "fluid_model_id_j", "system_exact_tie", "pair_fate"
    ]]
    pairs = pairs.merge(
        baseline,
        on=[*keys, "fluid_model_id_i", "fluid_model_id_j"],
        how="left",
        validate="one_to_one",
    )
    pairs["baseline_tie_match"] = pairs["system_exact_tie"].astype(bool).eq(pairs["same_integer"])
    order = [
        "group_id", "pair_id", *keys, "fluid_model_id_i", "fluid_name_i", "fluid_model_id_j", "fluid_name_j",
        "N_E_cont_i", "N_E_cont_j", "N_P_cont_i", "N_P_cont_j", "N_cont_i", "N_cont_j",
        "N_min_cont", "N_max_cont", "delta_N_cont", "next_integer_distance", "integer_boundary_margin",
        "Nstar_i", "Nstar_j", "same_integer", "delta_Nstar", "boundary_predicts_same_integer",
        "boundary_identity_match", "baseline_tie_match", "governing_constraint_i", "governing_constraint_j",
        "A_fluid_log", "A_filled_log", "A_dynamic_log", "A_system_log", "R_system_fluid",
        "N_mid_cont", "one_module_relative_resolution", "system_exact_tie", "pair_fate",
    ]
    return pairs[order].sort_values([*keys, "fluid_model_id_i", "fluid_model_id_j"]).reset_index(drop=True)


def _summary_record(part: pd.DataFrame, scope: str, label: str) -> dict[str, Any]:
    return {
        "summary_scope": scope,
        "summary_label": label,
        "n_pairs": int(len(part)),
        "n_configuration_equivalent": int(part["same_integer"].sum()),
        "configuration_equivalent_fraction": float(part["same_integer"].mean()),
        "median_R_system_fluid": float(part["R_system_fluid"].median()),
        "median_abs_A_system_log": float(part["A_system_log"].abs().median()),
        "median_delta_N_cont": float(part["delta_N_cont"].median()),
        "median_integer_boundary_margin": float(part["integer_boundary_margin"].median()),
        "minimum_integer_boundary_margin": float(part["integer_boundary_margin"].min()),
        "maximum_integer_boundary_margin": float(part["integer_boundary_margin"].max()),
    }


def build_design_consequence_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = [_summary_record(pairwise, "overall", "all")]
    for scenario, part in pairwise.groupby("scenario_id", sort=True):
        rows.append(_summary_record(part, "service", str(scenario)))
    for (fluid_i, fluid_j), part in pairwise.groupby(["fluid_model_id_i", "fluid_model_id_j"], sort=True):
        rows.append(_summary_record(part, "fluid_pair", f"{fluid_i} vs {fluid_j}"))
    granularity = pairwise.copy()
    granularity["granularity_bin"] = pd.qcut(
        granularity["one_module_relative_resolution"],
        q=5,
        labels=["finest", "fine", "intermediate", "coarse", "coarsest"],
        duplicates="drop",
    )
    for label, part in granularity.groupby("granularity_bin", observed=True, sort=False):
        record = _summary_record(part, "architecture_granularity", str(label))
        record["median_one_module_relative_resolution"] = float(part["one_module_relative_resolution"].median())
        rows.append(record)
    for (scenario, label), part in granularity.groupby(["scenario_id", "granularity_bin"], observed=True, sort=True):
        record = _summary_record(part, "service_by_granularity", f"{scenario}::{label}")
        record["median_one_module_relative_resolution"] = float(part["one_module_relative_resolution"].median())
        rows.append(record)
    return pd.DataFrame(rows)


def build_qc_report(
    continuous: pd.DataFrame,
    pairwise: pd.DataFrame,
    baseline_service: pd.DataFrame,
    baseline_pairwise: pd.DataFrame,
    baseline_hashes: dict[str, str],
    current_hashes: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    old_power = (
        baseline_service["power_limited_unit_count"].astype(int)
        > baseline_service["resolved_energy_limited_unit_count"].astype(int)
    )
    constraint_cross_tab = pd.crosstab(
        continuous["old_governing_constraint"], continuous["governing_constraint"]
    ).to_dict()
    root_limit = float(config["design_consequence_boundary"]["root_relative_energy_tolerance"])
    gates = {
        "Q1_integer_reproduction": {
            "pass": bool(len(continuous) == len(baseline_service) and continuous["N_star_match"].all()),
            "n_rows": int(len(continuous)),
            "n_matches": int(continuous["N_star_match"].sum()),
        },
        "Q2_constraint_mapping": {
            "pass": bool(
                len(continuous) == len(baseline_service)
                and int(old_power.sum()) == int((continuous["old_governing_constraint"] == "power_rating_constrained").sum())
            ),
            "baseline_energy_constrained": int((~old_power).sum()),
            "baseline_power_rating_constrained": int(old_power.sum()),
            "continuous_constraint_cross_tab": constraint_cross_tab,
        },
        "Q3_pairwise_regression": {
            "pass": bool(
                len(pairwise) == len(baseline_pairwise)
                and pairwise["baseline_tie_match"].all()
                and int(pairwise["same_integer"].sum()) == int(baseline_pairwise["system_exact_tie"].sum())
            ),
            "n_pairs": int(len(pairwise)),
            "n_equivalent": int(pairwise["same_integer"].sum()),
            "n_baseline_exact_ties": int(baseline_pairwise["system_exact_tie"].sum()),
            "n_groups_with_equivalence": int(pairwise.groupby("group_id")["same_integer"].any().sum()),
        },
        "Q4_boundary_identity": {
            "pass": bool(pairwise["boundary_identity_match"].all()),
            "n_matches": int(pairwise["boundary_identity_match"].sum()),
            "n_pairs": int(len(pairwise)),
        },
        "Q5_root_residual": {
            "pass": bool((continuous["root_relative_residual"] <= root_limit).all()),
            "registered_limit": root_limit,
            "maximum": float(continuous["root_relative_residual"].max()),
            "median": float(continuous["root_relative_residual"].median()),
        },
        "Q6_frozen_baseline_hashes": {
            "pass": bool(baseline_hashes == current_hashes),
            "baseline_hashes": baseline_hashes,
            "current_hashes": current_hashes,
        },
        "Q7_granularity_is_hypothesis_test": {
            "pass": True,
            "registered_as_acceptance_gate": False,
            "interpretation": "Monotonicity is not required; the observed service- and geometry-dependent response is reported without threshold engineering.",
        },
    }
    return {
        "status": "PASS" if all(item["pass"] for item in gates.values()) else "FAIL",
        "version": "18.0.0",
        "analysis_scope": "incremental continuous closure and integer design-consequence boundary on the frozen V16 physical baseline",
        "gates": gates,
    }


def write_qc_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / QC_JSON_FILENAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "V18 CONTINUOUS-CLOSURE AND DESIGN-CONSEQUENCE QC",
        f"Overall status: {report['status']}",
        f"Scope: {report['analysis_scope']}",
        "",
    ]
    for gate, detail in report["gates"].items():
        lines.append(f"{gate}: {'PASS' if detail['pass'] else 'FAIL'}")
        for key, value in detail.items():
            if key != "pass":
                lines.append(f"  {key}: {value}")
    (output_dir / QC_TEXT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_result_hashes(results_dir: str | Path, names: Iterable[str]) -> dict[str, str]:
    root = Path(results_dir)
    return {name: sha256_file(root / name) for name in names}
