"""V16 evidence-gated multiscale co-design and self-consistent service layer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .fluid_properties import fluid_properties, get_model_spec
from .ltne_v16 import BedGeometryV16, _layer_arrays, evaluate_packed_bed_v16
from .validation_v16 import run_dlr_external_validation, run_grid_and_energy_verification, run_particle_hierarchy


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out


def generate_geometry_pool(config: dict[str, Any], n: int, seed: int) -> pd.DataFrame:
    keys = ["internal_diameter_m", "bed_height_m", "porosity", "particle_diameter_m", "nominal_thermal_input_kW"]
    unit = _lhs(int(n), len(keys), int(seed))
    rows: list[dict[str, Any]] = []
    for i in range(int(n)):
        base: dict[str, Any] = {"base_design_id": f"G{i+1:04d}", "sample_seed": int(seed), "sample_index": i}
        for j, key in enumerate(keys):
            lo, hi = map(float, config["design_space"][key])
            base[key] = lo + unit[i, j] * (hi - lo)
        for ti, topo in enumerate(config["topologies"]):
            row = dict(base)
            row.update(topo)
            row["topology_id"] = f"T{ti+1}_{topo['topology']}_{topo['gradient_orientation']}"
            row["design_id"] = f"{base['base_design_id']}_{row['topology_id']}"
            rows.append(row)
    return pd.DataFrame(rows)


def _load_pair_gate(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    rel = Path(config["admissibility"]["pair_gate_file"])
    path = (config_path.parent.parent / rel).resolve()
    gate = pd.read_csv(path)
    gate["physics_comparison_eligible"] = gate["physics_comparison_eligible"].astype(str).str.lower().eq("true")
    gate["engineering_deployment_eligible"] = gate["engineering_deployment_eligible"].astype(str).str.lower().eq("true")
    return gate


def _admissibility(record: dict[str, Any], config: dict[str, Any], pair_row: dict[str, Any]) -> tuple[bool, str]:
    reasons: list[str] = []
    a = config["admissibility"]
    if not bool(pair_row["physics_comparison_eligible"]):
        reasons.append("pair_evidence_gate_failed")
    if not bool(record["correlation_valid"]):
        reasons.append("transport_correlation_domain_failed")
    if float(record["bed_to_particle_ratio_min"]) < float(a["minimum_bed_to_particle_ratio"]):
        reasons.append("bed_to_particle_ratio_below_minimum")
    if float(record["particle_Re_max"]) > float(a["maximum_particle_Re"]):
        reasons.append("particle_Re_above_maximum")
    max_balance = float(a["maximum_absolute_energy_balance_error_pct"])
    if abs(float(record["charge_energy_balance_error_pct"])) > max_balance:
        reasons.append("charge_energy_balance_failed")
    if abs(float(record["discharge_energy_balance_error_pct"])) > max_balance:
        reasons.append("discharge_energy_balance_failed")
    tol = float(a["temperature_bound_tolerance_K"])
    cold = float(record["cold_temperature_C"]) + 273.15
    hot = float(record["hot_temperature_C"]) + 273.15
    if float(record["temperature_min_K"]) < cold - tol or float(record["temperature_max_K"]) > hot + tol:
        reasons.append("temperature_bound_or_overshoot_failed")
    if bool(record["temperature_clipping_used"]):
        reasons.append("temperature_clipping_prohibited")
    if record["charge_t90_h"] is None or not math.isfinite(float(record["charge_t90_h"])):
        reasons.append("charge_t90_unresolved")
    if float(record["high_grade_discharge_fraction"]) < float(a["minimum_high_grade_discharge_fraction"]):
        reasons.append("high_grade_fraction_below_minimum")
    return len(reasons) == 0, "PASS" if not reasons else ";".join(reasons)


def _registered_ambient_temperature_K(record: dict[str, Any], config: dict[str, Any]) -> float | None:
    """Return an explicit loss-sink temperature, or ``None`` for an adiabatic run.

    A nonzero volumetric wall-loss coefficient is only meaningful with a
    registered sink.  The unit table carries that boundary forward so service
    reclosure, refinement, and scale checks cannot silently revert to a cold
    process boundary.
    """
    loss = float(record.get("wall_loss_W_m3K", 0.0))
    ambient_C = record.get("ambient_temperature_C")
    if ambient_C is None or (isinstance(ambient_C, float) and not math.isfinite(ambient_C)):
        if loss > 0.0:
            raise ValueError("nonzero wall loss requires registered ambient_temperature_C")
        return None
    return float(ambient_C) + 273.15


def evaluate_geometry_pool(
    config_path: str | Path,
    config: dict[str, Any],
    pool: pd.DataFrame,
    n_cells: int,
    property_multipliers_by_model: dict[str, dict[str, float]] | None = None,
    model_ids: list[str] | None = None,
    physics_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    # Geometries are independent model evaluations.  Dispatch batches rather
    # than individual rows to limit Windows process-serialization overhead and
    # to preserve the original geometry-major/model-minor output order.
    execution = config.get("execution", {})
    workers = int(execution.get("geometry_parallel_workers", 1))
    batch_size = int(execution.get("geometry_parallel_geometry_batch_size", 1))
    if workers > 1 and len(pool) > 1:
        from concurrent.futures import ProcessPoolExecutor

        records = pool.to_dict("records")
        batches = [
            records[index:index + max(1, batch_size)]
            for index in range(0, len(records), max(1, batch_size))
        ]
        tasks = [
            (
                str(config_path),
                batch,
                config,
                int(n_cells),
                property_multipliers_by_model,
                model_ids,
                physics_overrides,
            )
            for batch in batches
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            frames = list(executor.map(_evaluate_geometry_batch, tasks))
        return pd.concat(frames, ignore_index=True)

    config_path = Path(config_path)
    gate = _load_pair_gate(config_path, config).set_index("fluid_model_id")
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    dT = hot_K - cold_K
    solid = config["reference_geometry"]
    if physics_overrides and "ambient_temperature_C" in physics_overrides:
        ambient_temperature_C = float(physics_overrides["ambient_temperature_C"])
    elif "ambient_temperature_C" in config.get("model_form_stress", {}):
        ambient_temperature_C = float(config["model_form_stress"]["ambient_temperature_C"])
    else:
        ambient_temperature_C = None
    rows: list[dict[str, Any]] = []
    for geom_row in pool.to_dict("records"):
        geom = BedGeometryV16(
            internal_diameter_m=float(geom_row["internal_diameter_m"]),
            bed_height_m=float(geom_row["bed_height_m"]),
            porosity=float(geom_row["porosity"]),
            particle_diameter_m=float(geom_row["particle_diameter_m"]),
            topology=str(geom_row["topology"]),
            gradient_ratio=float(geom_row["gradient_ratio"]),
            gradient_orientation=str(geom_row["gradient_orientation"]),
            layer_fraction=float(geom_row["layer_fraction"]),
        )
        external_area_to_volume = 4.0 / geom.internal_diameter_m + 2.0 / geom.bed_height_m
        surface_u = float((physics_overrides or {}).get("surface_heat_transfer_coefficient_W_m2K", 0.0))
        if "surface_heat_transfer_coefficient_W_m2K" in (physics_overrides or {}):
            wall_loss_W_m3K = surface_u * external_area_to_volume
        else:
            wall_loss_W_m3K = float((physics_overrides or {}).get("wall_loss_W_m3K", 0.0))
        for model_id in (config["fluid_model_ids"] if model_ids is None else model_ids):
            cp = fluid_properties(model_id, 0.5 * (cold_K + hot_K)).cp_J_kgK
            flow = float(geom_row["nominal_thermal_input_kW"]) * 1000.0 / (cp * dT)
            multipliers = None if not property_multipliers_by_model else property_multipliers_by_model.get(model_id)
            result = evaluate_packed_bed_v16(
                model_id=model_id,
                design_id=f"{geom_row['design_id']}__{model_id}",
                geometry=geom,
                mass_flow_kg_s=flow,
                cold_temperature_K=cold_K,
                hot_temperature_K=hot_K,
                solid_density_kg_m3=float(solid["solid_density_kg_m3"]),
                solid_cp_J_kgK=float(solid["solid_cp_J_kgK"]),
                solid_thermal_conductivity_W_mK=float(solid["solid_thermal_conductivity_W_mK"]),
                n_cells=int(n_cells),
                axial_dispersion_m2_s=float((physics_overrides or {}).get("axial_dispersion_m2_s", 0.0)),
                solid_axial_diffusivity_m2_s=float((physics_overrides or {}).get("solid_axial_diffusivity_m2_s", 0.0)),
                wall_loss_W_m3K=wall_loss_W_m3K,
                ambient_temperature_K=(
                    None if ambient_temperature_C is None else ambient_temperature_C + 273.15
                ),
                minimum_outlet_grade_fraction=float(config["service_scenarios"][0]["minimum_outlet_grade_fraction"]),
                property_multipliers=multipliers,
            )
            result.update({k: geom_row[k] for k in ("base_design_id", "topology_id", "sample_seed", "sample_index")})
            result["surface_heat_transfer_coefficient_W_m2K"] = surface_u
            result["external_area_to_packed_volume_ratio_m1"] = external_area_to_volume
            result["ambient_temperature_C"] = ambient_temperature_C
            pair = gate.loc[model_id].to_dict()
            result.update({
                "skeleton_id": solid["skeleton_id"],
                "pair_evidence_tier": pair["evidence_tier"],
                "pair_chemical_compatibility_status": pair["chemical_compatibility_status"],
                "engineering_deployment_eligible": bool(pair["engineering_deployment_eligible"]),
            })
            ok, reason = _admissibility(result, config, pair)
            result["physics_comparison_admissible"] = ok
            result["admissibility_reason"] = reason
            rows.append(result)
    return pd.DataFrame(rows)


def _evaluate_geometry_batch(
    task: tuple[
        str,
        list[dict[str, Any]],
        dict[str, Any],
        int,
        dict[str, dict[str, float]] | None,
        list[str] | None,
        dict[str, float] | None,
    ],
) -> pd.DataFrame:
    """Evaluate a deterministic geometry batch in one worker process."""
    config_path, records, config, n_cells, multipliers, model_ids, overrides = task
    local_config = deepcopy(config)
    local_config.setdefault("execution", {})["geometry_parallel_workers"] = 1
    return evaluate_geometry_pool(
        config_path=Path(config_path),
        config=local_config,
        pool=pd.DataFrame(records),
        n_cells=n_cells,
        property_multipliers_by_model=multipliers,
        model_ids=model_ids,
        physics_overrides=overrides,
    )



def screen_confirmation(screen: pd.DataFrame, confirmation: pd.DataFrame) -> pd.DataFrame:
    """Quantify low-fidelity screening error against the production confirmation run."""
    metrics = [
        "charge_t90_h", "high_grade_discharge_fraction",
        "thermal_front_thickness_ratio", "pressure_drop_kPa",
        "volumetric_capacity_MWh_m3",
    ]
    left = screen[["design_id", "n_cells"] + metrics].rename(
        columns={"n_cells": "screen_n_cells", **{m: f"screen_{m}" for m in metrics}}
    )
    right = confirmation[["design_id", "n_cells"] + metrics].rename(
        columns={"n_cells": "confirmation_n_cells", **{m: f"confirmation_{m}" for m in metrics}}
    )
    out = left.merge(right, on="design_id", how="inner")
    for m in metrics:
        a = out[f"screen_{m}"].to_numpy(float)
        b = out[f"confirmation_{m}"].to_numpy(float)
        out[f"{m}_absolute_difference"] = a - b
        out[f"{m}_relative_difference_pct"] = 100.0 * (a - b) / np.maximum(np.abs(b), 1e-12)
    return out

def pareto_mask(frame: pd.DataFrame, objectives: list[list[str]] | list[tuple[str, str]]) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = np.empty((len(frame), len(objectives)), dtype=float)
    for j, (key, direction) in enumerate(objectives):
        v = frame[key].to_numpy(float)
        values[:, j] = -v if direction == "maximize" else v
    keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        if not keep[i]:
            continue
        dominated = np.all(values <= values[i] + 1e-14, axis=1) & np.any(values < values[i] - 1e-14, axis=1)
        if np.any(dominated):
            keep[i] = False
    return keep


def unit_pareto(results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    valid = results[results["physics_comparison_admissible"]].reset_index(drop=True)
    mask = pareto_mask(valid, config["unit_objectives"])
    front = valid.loc[mask].copy()
    front["pareto_scope"] = "pooled_material_topology_unit_reference"
    return front


def _pressure_drop_actual(
    *, model_id: str, geometry: BedGeometryV16, mass_flow_kg_s: float,
    mean_temperature_K: float, n_cells: int = 48,
    multipliers: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    p = fluid_properties(model_id, mean_temperature_K)
    mult = {"density": 1.0, "viscosity": 1.0}
    if multipliers:
        for k in mult:
            mult[k] = float(multipliers.get(k, 1.0))
    rho = p.density_kg_m3 * mult["density"]
    mu = p.viscosity_Pa_s * mult["viscosity"]
    area = math.pi * geometry.internal_diameter_m**2 / 4.0
    eps, dp = _layer_arrays(geometry, n_cells)
    u = mass_flow_kg_s / (rho * area)
    dz = geometry.bed_height_m / n_cells
    viscous = 0.0
    inertial = 0.0
    for e, d in zip(eps, dp):
        viscous += dz * 150.0 * mu * u * (1.0 - e) ** 2 / (e**3 * d**2)
        inertial += dz * 1.75 * rho * u**2 * (1.0 - e) / (e**3 * d)
    return viscous + inertial, viscous, inertial


def _darcy_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Return a Darcy friction factor without an unregistered constant K.

    Laminar flow uses the analytic result.  The turbulent branch is the
    explicit Swamee--Jain approximation to the Colebrook relation.  A simple
    laminar--turbulent switch is retained for transparency, so this bounded
    friction screen is not a substitute for a commissioned manifold design.
    """
    re = max(float(reynolds), 1.0e-12)
    if re < 2300.0:
        return 64.0 / re
    rr = max(float(relative_roughness), 0.0)
    term = rr / 3.7 + 5.74 / (re ** 0.9)
    return 0.25 / (math.log10(max(term, 1.0e-20)) ** 2)


def _pipe_friction_drop_Pa(
    volumetric_flow_m3_s: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    diameter_m: float,
    length_m: float,
    roughness_m: float,
) -> float:
    """Darcy--Weisbach loss for one full circular-pipe segment."""
    if volumetric_flow_m3_s <= 0.0 or length_m <= 0.0:
        return 0.0
    area = math.pi * diameter_m**2 / 4.0
    velocity = volumetric_flow_m3_s / area
    reynolds = density_kg_m3 * velocity * diameter_m / max(viscosity_Pa_s, 1.0e-20)
    factor = _darcy_friction_factor(reynolds, roughness_m / diameter_m)
    return factor * length_m / diameter_m * 0.5 * density_kg_m3 * velocity**2


def _array_header_hydraulics(
    *,
    parallel_unit_count: int,
    total_volumetric_flow_m3_s: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    module_diameter_m: float,
    config: dict[str, Any],
) -> dict[str, float | int | str]:
    """Screen the friction of a balanced two-level modular header network.

    Each module receives an equal share of the service flow.  Modules are
    arranged in a near-square array; a trunk pair feeds lateral header pairs.
    Diameters are sized from the registered maximum header velocity.  The
    returned pressure is the largest frictional route (trunk plus lateral,
    supply and return).  This removes the former geometry-independent loss
    coefficient while retaining a deliberately narrow, auditable scope.
    """
    if parallel_unit_count < 1 or total_volumetric_flow_m3_s <= 0.0:
        raise ValueError("array-header screen requires positive count and flow")
    hcfg = config["service_model"]["array_header_hydraulics"]
    design_velocity = float(hcfg["header_design_velocity_m_s"])
    roughness = float(hcfg["pipe_absolute_roughness_m"])
    pitch = float(hcfg["module_pitch_multiplier"]) * float(module_diameter_m)
    min_length = float(hcfg["minimum_header_length_m"])
    if design_velocity <= 0.0 or roughness < 0.0 or pitch <= 0.0 or min_length <= 0.0:
        raise ValueError("invalid registered array-header geometry")

    n_rows = max(1, int(math.floor(math.sqrt(parallel_unit_count))))
    n_columns = int(math.ceil(parallel_unit_count / n_rows))
    counts = [n_columns] * n_rows
    excess = n_rows * n_columns - parallel_unit_count
    for i in range(excess):
        counts[-(i + 1)] -= 1
    counts = [count for count in counts if count > 0]
    n_rows = len(counts)
    unit_flow = total_volumetric_flow_m3_s / parallel_unit_count

    lateral_losses: list[float] = []
    lateral_diameters: list[float] = []
    for count in counts:
        row_flow = count * unit_flow
        diameter = math.sqrt(4.0 * row_flow / (math.pi * design_velocity))
        segment_length = max(min_length, count * pitch) / count
        one_direction = 0.0
        for segment in range(count):
            segment_flow = row_flow - segment * unit_flow
            one_direction += _pipe_friction_drop_Pa(
                segment_flow, density_kg_m3, viscosity_Pa_s, diameter,
                segment_length, roughness,
            )
        lateral_losses.append(2.0 * one_direction)
        lateral_diameters.append(diameter)

    trunk_diameter = math.sqrt(4.0 * total_volumetric_flow_m3_s / (math.pi * design_velocity))
    trunk_segment_length = max(min_length, n_rows * pitch) / n_rows
    trunk_one_direction = 0.0
    remaining = total_volumetric_flow_m3_s
    for count in counts:
        trunk_one_direction += _pipe_friction_drop_Pa(
            remaining, density_kg_m3, viscosity_Pa_s, trunk_diameter,
            trunk_segment_length, roughness,
        )
        remaining -= count * unit_flow
    trunk_pair = 2.0 * trunk_one_direction
    lateral_pair = max(lateral_losses)
    total = trunk_pair + lateral_pair
    return {
        "header_pressure_drop_Pa": total,
        "header_trunk_pressure_drop_Pa": trunk_pair,
        "header_lateral_pressure_drop_Pa": lateral_pair,
        "header_array_rows": n_rows,
        "header_array_max_columns": max(counts),
        "header_module_pitch_m": pitch,
        "header_trunk_diameter_m": trunk_diameter,
        "header_lateral_diameter_min_m": min(lateral_diameters),
        "header_lateral_diameter_max_m": max(lateral_diameters),
        "header_design_velocity_m_s": design_velocity,
        "header_hydraulic_model": str(hcfg["model"]),
    }


def size_fixed_service(results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Size each service case at its resolved per-unit flow.

    Unit count changes the flow seen by every module. That flow changes heat
    transfer, high-grade energy and pressure loss. V16 therefore resolves the
    integer unit count and the LTNE response together. The returned design is
    checked at the final flow and at the adjacent lower integer count.
    """
    # Each (service, unit-design) closure is independent.  The transient
    # calculation itself is deterministic, whereas evaluating the thousands
    # of closures serially only makes the reproducible build unnecessarily
    # slow.  Parallelism is therefore restricted to this outer map; row order
    # remains scenario-major and design-major through ``executor.map``.
    execution = config.get("execution", {})
    workers = int(execution.get("service_parallel_workers", 1))
    chunk_size = int(execution.get("service_parallel_chunk_size", 1))
    if workers > 1 and len(results) > 1 and len(config["service_scenarios"]) > 0:
        from concurrent.futures import ProcessPoolExecutor

        records = results.to_dict("records")
        tasks = [
            (record, scenario, config)
            for scenario in config["service_scenarios"]
            for record in records
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(
                executor.map(
                    _size_fixed_service_record,
                    tasks,
                    chunksize=max(1, chunk_size),
                )
            )
        return pd.DataFrame(rows)

    rows: list[dict[str, Any]] = []
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    dT = hot_K - cold_K
    svc_cfg = config["service_model"]
    solid = config["reference_geometry"]
    max_iter = int(svc_cfg.get("closure_max_iterations", 20))
    for scenario in config["service_scenarios"]:
        power_MW = float(scenario["target_thermal_power_MW"])
        duration_h = float(scenario["storage_duration_h"])
        required_energy = power_MW * duration_h
        for r in results.to_dict("records"):
            mult = {
                "density": float(r.get("density_multiplier", 1.0)),
                "cp": float(r.get("cp_multiplier", 1.0)),
                "viscosity": float(r.get("viscosity_multiplier", 1.0)),
                "conductivity": float(r.get("conductivity_multiplier", 1.0)),
            }
            transport_temperature = scenario.get("transport_property_temperature_K")
            if transport_temperature is None:
                cp = (
                    float(r["mean_cp_J_kgK"])
                    if "mean_cp_J_kgK" in r
                    else fluid_properties(r["fluid_model_id"], 0.5 * (cold_K + hot_K)).cp_J_kgK * mult["cp"]
                )
            else:
                # Keep the imposed service power exact when the run-wise
                # frozen-property temperature changes.  The endpoint test
                # changes cp as well as rho, mu and k, so its total mass flow
                # must be recomputed from P = m_dot cp DeltaT.
                cp = fluid_properties(
                    str(r["fluid_model_id"]), float(transport_temperature)
                ).cp_J_kgK * mult["cp"]
            unit_power_MW = float(r["nominal_thermal_input_kW"]) / 1000.0
            effective_unit_energy = float(r["stored_heat_MWh"]) * float(r["high_grade_discharge_fraction"])
            n_power = int(math.ceil(power_MW / max(unit_power_MW, 1e-12)))
            n_energy_nominal = int(math.ceil(required_energy / max(effective_unit_energy, 1e-12)))
            n_units = max(1, n_power, n_energy_nominal)
            minimum_duty_ratio = float(svc_cfg.get("minimum_resolved_to_nominal_duty_ratio", 0.05))
            duty_limited_max_count = int(math.floor(power_MW / max(minimum_duty_ratio * unit_power_MW, 1e-12)))
            evaluation_count_limit = max(n_power, min(int(svc_cfg["maximum_unit_count"]), duty_limited_max_count))
            initial_count_exceeded_limit = n_units > evaluation_count_limit
            if initial_count_exceeded_limit:
                n_units = n_power
            total_flow = power_MW * 1e6 / (cp * dT)
            geometry = BedGeometryV16(
                internal_diameter_m=float(r["internal_diameter_m"]),
                bed_height_m=float(r["bed_height_m"]),
                porosity=float(r["porosity"]),
                particle_diameter_m=float(r["particle_diameter_m"]),
                topology=str(r["topology"]),
                gradient_ratio=float(r["gradient_ratio"]),
                gradient_orientation=str(r["gradient_orientation"]),
                layer_fraction=float(r["layer_fraction"]),
            )
            cache: dict[int, dict[str, Any]] = {}

            def resolve_at_count(count: int) -> dict[str, Any]:
                count = max(1, int(count))
                if count not in cache:
                    unit_flow = total_flow / count
                    actual = evaluate_packed_bed_v16(
                        model_id=str(r["fluid_model_id"]),
                        design_id=f"{r['design_id']}__{scenario['scenario_id']}__N{count}",
                        geometry=geometry,
                        mass_flow_kg_s=unit_flow,
                        cold_temperature_K=cold_K,
                        hot_temperature_K=hot_K,
                        solid_density_kg_m3=float(solid["solid_density_kg_m3"]),
                        solid_cp_J_kgK=float(solid["solid_cp_J_kgK"]),
                        solid_thermal_conductivity_W_mK=float(solid["solid_thermal_conductivity_W_mK"]),
                        n_cells=int(r["n_cells"]),
                        axial_dispersion_m2_s=float(r.get("axial_dispersion_m2_s", 0.0)),
                        solid_axial_diffusivity_m2_s=float(r.get("solid_axial_diffusivity_m2_s", 0.0)),
                        wall_loss_W_m3K=float(r.get("wall_loss_W_m3K", 0.0)),
                        ambient_temperature_K=_registered_ambient_temperature_K(r, config),
                        minimum_outlet_grade_fraction=float(scenario["minimum_outlet_grade_fraction"]),
                        property_multipliers=mult,
                        transport_property_temperature_K=(
                            None if scenario.get("transport_property_temperature_K") is None
                            else float(scenario["transport_property_temperature_K"])
                        ),
                    )
                    actual_energy = float(actual["stored_heat_MWh"]) * float(actual["high_grade_discharge_fraction"])
                    actual["required_energy_unit_count"] = int(math.ceil(required_energy / max(actual_energy, 1e-12)))
                    cache[count] = actual
                return cache[count]

            visited: set[int] = set()
            closure_status = "ITERATION_LIMIT"
            closure_solver_feasible = True
            projected_required_unit_count = n_units
            for _ in range(max_iter):
                if n_units in visited:
                    n_units = max(visited)
                    closure_status = "INTEGER_CYCLE_RESOLVED_CONSERVATIVELY"
                    break
                visited.add(n_units)
                trial = resolve_at_count(n_units)
                required_count = max(n_power, int(trial["required_energy_unit_count"]))
                projected_required_unit_count = required_count
                if required_count > evaluation_count_limit:
                    closure_status = "DUTY_OR_UNIT_COUNT_LIMIT_EXCEEDED"
                    closure_solver_feasible = False
                    break
                if required_count == n_units:
                    closure_status = "FIXED_POINT"
                    break
                n_units = required_count

            def installed_capacity(count: int) -> tuple[float, dict[str, Any]]:
                resolved = resolve_at_count(count)
                capacity = count * float(resolved["stored_heat_MWh"]) * float(resolved["high_grade_discharge_fraction"])
                return capacity, resolved

            installed_high_grade, actual = installed_capacity(n_units)
            if closure_solver_feasible and installed_high_grade + 1e-12 < required_energy:
                for _ in range(max_iter):
                    next_count = max(n_units + 1, n_power, int(actual["required_energy_unit_count"]))
                    projected_required_unit_count = next_count
                    if next_count > evaluation_count_limit:
                        closure_status = "DUTY_OR_UNIT_COUNT_LIMIT_EXCEEDED"
                        closure_solver_feasible = False
                        break
                    n_units = next_count
                    installed_high_grade, actual = installed_capacity(n_units)
                    if installed_high_grade + 1e-12 >= required_energy:
                        closure_status = "FEASIBLE_UPPER_BRACKET"
                        break
                else:
                    closure_status = "FEASIBLE_UPPER_BRACKET_NOT_FOUND"
                    closure_solver_feasible = False

            # Find the smallest feasible integer count.  The lower boundary is
            # the power constraint.  The upper boundary is a capacity-feasible
            # count from the coupled LTNE evaluation.  The adjacent count is
            # always evaluated after the binary search, so integer minimality is
            # an audited result rather than an assumption.
            if closure_solver_feasible:
                power_capacity, power_actual = installed_capacity(n_power)
                if power_capacity + 1e-12 >= required_energy:
                    n_units = n_power
                    installed_high_grade, actual = power_capacity, power_actual
                else:
                    lower_infeasible = n_power
                    upper_feasible = n_units
                    while upper_feasible - lower_infeasible > 1:
                        middle = (lower_infeasible + upper_feasible) // 2
                        middle_capacity, _ = installed_capacity(middle)
                        if middle_capacity + 1e-12 >= required_energy:
                            upper_feasible = middle
                        else:
                            lower_infeasible = middle
                    n_units = upper_feasible
                    installed_high_grade, actual = installed_capacity(n_units)
                closure_status = "MINIMAL_INTEGER_COUNT_BINARY_BRACKET"
                projected_required_unit_count = int(actual["required_energy_unit_count"])

            adjacent_lower_feasible = False
            if closure_solver_feasible and n_units > n_power:
                adjacent_lower_capacity, _ = installed_capacity(n_units - 1)
                adjacent_lower_feasible = adjacent_lower_capacity + 1e-12 >= required_energy
                if adjacent_lower_feasible:
                    closure_status = "NONMONOTONE_OR_UNRESOLVED_INTEGER_RESPONSE"
                    closure_solver_feasible = False

            unit_flow = total_flow / n_units
            bed_dp = float(actual["pressure_drop_kPa"]) * 1000.0
            _, visc_dp, iner_dp = _pressure_drop_actual(
                model_id=str(r["fluid_model_id"]), geometry=geometry, mass_flow_kg_s=unit_flow,
                mean_temperature_K=0.5*(cold_K+hot_K), n_cells=int(r["n_cells"]), multipliers=mult,
            )
            p_mean = fluid_properties(str(r["fluid_model_id"]), 0.5*(cold_K+hot_K))
            rho = p_mean.density_kg_m3 * mult["density"]
            header = _array_header_hydraulics(
                parallel_unit_count=n_units,
                total_volumetric_flow_m3_s=total_flow / rho,
                density_kg_m3=rho,
                viscosity_Pa_s=p_mean.viscosity_Pa_s * mult["viscosity"],
                module_diameter_m=geometry.internal_diameter_m,
                config=config,
            )
            header_dp = float(header["header_pressure_drop_Pa"])
            total_dp = bed_dp + header_dp
            pump_power = total_dp * (total_flow / rho)
            pump_energy_fraction = pump_power * duration_h * 3600.0 / max(required_energy * 3.6e9, 1e-12)
            installed_volume = n_units * float(r["storage_volume_m3"])
            salt_t = n_units * float(r["salt_inventory_kg"]) / 1000.0
            solid_t = n_units * float(r["solid_inventory_kg"]) / 1000.0
            resolved_duration = installed_high_grade / power_MW
            n_energy_actual = int(actual["required_energy_unit_count"])
            closure_error_pct = 100.0 * (installed_high_grade / required_energy - 1.0)
            actual_charge_t90 = float(actual["charge_t90_h"]) if actual["charge_t90_h"] is not None else float("nan")
            actual_discharge_t90 = float(actual["discharge_t90_h"]) if actual["discharge_t90_h"] is not None else float("nan")
            t90_denominator = actual_charge_t90 + actual_discharge_t90
            actual_balance_ok = (
                abs(float(actual["charge_energy_balance_error_pct"])) <= float(config["admissibility"]["maximum_absolute_energy_balance_error_pct"]) and
                abs(float(actual["discharge_energy_balance_error_pct"])) <= float(config["admissibility"]["maximum_absolute_energy_balance_error_pct"])
            )
            service_ok = (
                bool(r["physics_comparison_admissible"]) and
                actual_balance_ok and
                closure_solver_feasible and
                bed_dp / 1000.0 <= float(svc_cfg["maximum_unit_pressure_drop_kPa"]) and
                n_units <= int(svc_cfg["maximum_unit_count"]) and
                resolved_duration + 1e-12 >= duration_h and
                n_units * unit_power_MW + 1e-12 >= power_MW and
                not adjacent_lower_feasible
            )
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "target_thermal_power_MW": power_MW,
                "storage_duration_h": duration_h,
                "required_energy_MWh": required_energy,
                "design_id": r["design_id"],
                "base_design_id": r["base_design_id"],
                "topology_id": r["topology_id"],
                "fluid_model_id": r["fluid_model_id"],
                "fluid_name": r["fluid_name"],
                "topology": r["topology"],
                "gradient_orientation": r["gradient_orientation"],
                "pair_evidence_tier": r["pair_evidence_tier"],
                "engineering_deployment_eligible": r["engineering_deployment_eligible"],
                "parallel_unit_count": n_units,
                "power_limited_unit_count": n_power,
                "nominal_energy_limited_unit_count": n_energy_nominal,
                "resolved_energy_limited_unit_count": n_energy_actual,
                "projected_required_unit_count": projected_required_unit_count,
                "maximum_resolvable_unit_count": evaluation_count_limit,
                "minimum_resolved_to_nominal_duty_ratio": minimum_duty_ratio,
                "service_closure_solver_feasible": closure_solver_feasible,
                "resolved_unit_mass_flow_kg_s": unit_flow,
                "resolved_unit_thermal_power_kW": power_MW * 1000.0 / n_units,
                "resolved_to_nominal_unit_duty_ratio": (power_MW / n_units) / max(unit_power_MW, 1e-12),
                "nominal_high_grade_discharge_fraction": float(r["high_grade_discharge_fraction"]),
                "resolved_high_grade_discharge_fraction": float(actual["high_grade_discharge_fraction"]),
                "resolved_unit_deliverable_energy_MWh": float(actual["deliverable_energy_MWh"]),
                "minimum_outlet_grade_fraction": float(scenario["minimum_outlet_grade_fraction"]),
                "deliverable_energy_definition": actual["deliverable_energy_definition"],
                "cycle_coupled_discharge": bool(actual["cycle_coupled_discharge"]),
                "charge_final_inventory_fraction": float(actual["charge_final_inventory_fraction"]),
                "threshold_crossing_interpolation_used": bool(actual["threshold_crossing_interpolation_used"]),
                "transport_property_temperature_C": float(actual["transport_property_temperature_C"]),
                "transport_property_treatment": actual["transport_property_treatment"],
                "resolved_charge_t90_h": actual_charge_t90,
                "resolved_discharge_t90_h": actual_discharge_t90,
                "charge_discharge_t90_ratio": actual_charge_t90 / max(actual_discharge_t90, 1e-12),
                "charge_discharge_t90_asymmetry_pct": 200.0 * (
                    actual_charge_t90 - actual_discharge_t90
                ) / max(t90_denominator, 1e-12),
                "charge_discharge_wall_loss_difference_percentage_point": 100.0 * (
                    float(actual["charge_wall_loss_fraction"]) - float(actual["discharge_wall_loss_fraction"])
                ),
                "surface_heat_transfer_coefficient_W_m2K": float(r.get("surface_heat_transfer_coefficient_W_m2K", 0.0)),
                "external_area_to_packed_volume_ratio_m1": float(r.get("external_area_to_packed_volume_ratio_m1", 4.0 / geometry.internal_diameter_m + 2.0 / geometry.bed_height_m)),
                "wall_loss_W_m3K": float(actual["wall_loss_W_m3K"]),
                "service_solver_evaluations": len(cache),
                "service_count_search_method": "binary_monotone_bracket_with_adjacent_verification",
                "service_closure_status": closure_status,
                "adjacent_lower_count_feasible": adjacent_lower_feasible,
                "service_capacity_closure_error_pct": closure_error_pct,
                "installed_packed_volume_m3": installed_volume,
                "installed_high_grade_capacity_MWh": installed_high_grade,
                "resolved_high_grade_duration_h": resolved_duration,
                "service_compactness_MWh_m3": required_energy / installed_volume,
                "salt_inventory_t": salt_t,
                "solid_inventory_t": solid_t,
                "capacity_oversize_pct": closure_error_pct,
                "nominal_power_margin_pct": 100.0 * (n_units * unit_power_MW / power_MW - 1.0),
                "unit_bed_pressure_drop_kPa": bed_dp / 1000.0,
                "unit_viscous_pressure_drop_kPa": visc_dp / 1000.0,
                "unit_inertial_pressure_drop_kPa": iner_dp / 1000.0,
                "header_pressure_drop_kPa": header_dp / 1000.0,
                "header_trunk_pressure_drop_kPa": float(header["header_trunk_pressure_drop_Pa"]) / 1000.0,
                "header_lateral_pressure_drop_kPa": float(header["header_lateral_pressure_drop_Pa"]) / 1000.0,
                "header_array_rows": int(header["header_array_rows"]),
                "header_array_max_columns": int(header["header_array_max_columns"]),
                "header_module_pitch_m": float(header["header_module_pitch_m"]),
                "header_trunk_diameter_m": float(header["header_trunk_diameter_m"]),
                "header_lateral_diameter_min_m": float(header["header_lateral_diameter_min_m"]),
                "header_lateral_diameter_max_m": float(header["header_lateral_diameter_max_m"]),
                "header_design_velocity_m_s": float(header["header_design_velocity_m_s"]),
                "header_hydraulic_model": str(header["header_hydraulic_model"]),
                "system_pump_power_kW": pump_power / 1000.0,
                "pump_energy_fraction": pump_energy_fraction,
                "pump_energy_definition": "frictional energy for the modeled packed bed plus the registered balanced on-array trunk-and-lateral header screen; fittings, valves, heat exchangers, external piping and pump efficiency excluded",
                "service_admissible": service_ok,
                "model_scope": svc_cfg["model_scope"],
            })
    return pd.DataFrame(rows)


def _size_fixed_service_record(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one independent service/design closure in a worker process.

    The worker calls the serial branch explicitly so nested process pools are
    impossible.  A one-record DataFrame is used deliberately: it guarantees
    that the calculation follows exactly the same code path as the serial
    production closure.
    """
    record, scenario, config = task
    local_config = deepcopy(config)
    local_config["service_scenarios"] = [scenario]
    local_config.setdefault("execution", {})["service_parallel_workers"] = 1
    resolved = size_fixed_service(pd.DataFrame([record]), local_config)
    return resolved.iloc[0].to_dict()


def outlet_grade_threshold_sensitivity(
    unit: pd.DataFrame,
    baseline_service: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Re-close every service case at each registered outlet-grade threshold."""
    thresholds = [float(x) for x in config["verification"]["outlet_grade_thresholds"]]
    frames: list[pd.DataFrame] = []
    for threshold in thresholds:
        print(f"[V16 threshold] closing all service cases at g = {threshold:.2f}", flush=True)
        if math.isclose(threshold, 0.9, rel_tol=0.0, abs_tol=1e-12):
            frame = baseline_service.copy()
        else:
            threshold_config = deepcopy(config)
            for scenario in threshold_config["service_scenarios"]:
                scenario["minimum_outlet_grade_fraction"] = threshold
            frame = size_fixed_service(unit, threshold_config)
        frame["threshold_sensitivity_grade_fraction"] = threshold
        frames.append(frame)
    sensitivity = pd.concat(frames, ignore_index=True)
    keys = ["scenario_id", "design_id"]
    baseline = sensitivity[np.isclose(sensitivity["threshold_sensitivity_grade_fraction"], 0.9)][
        keys + ["parallel_unit_count", "installed_packed_volume_m3", "resolved_unit_deliverable_energy_MWh"]
    ].rename(columns={
        "parallel_unit_count": "g090_parallel_unit_count",
        "installed_packed_volume_m3": "g090_installed_packed_volume_m3",
        "resolved_unit_deliverable_energy_MWh": "g090_resolved_unit_deliverable_energy_MWh",
    })
    sensitivity = sensitivity.merge(baseline, on=keys, how="left", validate="many_to_one")
    sensitivity["parallel_unit_count_ratio_vs_g090"] = (
        sensitivity["parallel_unit_count"] / sensitivity["g090_parallel_unit_count"]
    )
    sensitivity["installed_volume_ratio_vs_g090"] = (
        sensitivity["installed_packed_volume_m3"] / sensitivity["g090_installed_packed_volume_m3"]
    )
    sensitivity["deliverable_energy_ratio_vs_g090"] = (
        sensitivity["resolved_unit_deliverable_energy_MWh"]
        / sensitivity["g090_resolved_unit_deliverable_energy_MWh"]
    )
    sensitivity["interpretation"] = (
        "Deterministic registered-threshold sensitivity; not a probability distribution or chemistry-validation uncertainty."
    )
    return sensitivity


def _module_scale_case(
    task: tuple[dict[str, Any], dict[str, Any], float, list[int], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one independent retained-design/module-scale case."""
    source, scenario, factor, ceilings, config = task
    solid = config["reference_geometry"]
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    # Preserve the reference axial cell length so that an apparent module-scale
    # response cannot be caused by a coarser grid.
    scaled_n_cells = max(8, int(round(int(source["n_cells"]) * factor)))
    geometry = BedGeometryV16(
        internal_diameter_m=float(source["internal_diameter_m"]) * factor,
        bed_height_m=float(source["bed_height_m"]) * factor,
        porosity=float(source["porosity"]),
        particle_diameter_m=float(source["particle_diameter_m"]),
        topology=str(source["topology"]),
        gradient_ratio=float(source["gradient_ratio"]),
        gradient_orientation=str(source["gradient_orientation"]),
        layer_fraction=float(source["layer_fraction"]),
    )
    scaled_id = f"{source['design_id']}__MODULE_SCALE_{factor:g}"
    evaluated = evaluate_packed_bed_v16(
        model_id=str(source["fluid_model_id"]), design_id=scaled_id, geometry=geometry,
        mass_flow_kg_s=float(source["mass_flow_kg_s"]) * factor**2,
        cold_temperature_K=cold_K, hot_temperature_K=hot_K,
        solid_density_kg_m3=float(solid["solid_density_kg_m3"]),
        solid_cp_J_kgK=float(solid["solid_cp_J_kgK"]),
        solid_thermal_conductivity_W_mK=float(solid["solid_thermal_conductivity_W_mK"]),
        n_cells=scaled_n_cells,
        axial_dispersion_m2_s=float(source.get("axial_dispersion_m2_s", 0.0)),
        solid_axial_diffusivity_m2_s=float(source.get("solid_axial_diffusivity_m2_s", 0.0)),
        wall_loss_W_m3K=float(source.get("wall_loss_W_m3K", 0.0)),
        ambient_temperature_K=_registered_ambient_temperature_K(source, config),
        minimum_outlet_grade_fraction=float(scenario["minimum_outlet_grade_fraction"]),
        property_multipliers={
            "density": float(source.get("density_multiplier", 1.0)),
            "cp": float(source.get("cp_multiplier", 1.0)),
            "viscosity": float(source.get("viscosity_multiplier", 1.0)),
            "conductivity": float(source.get("conductivity_multiplier", 1.0)),
        },
    )
    scaled_record = dict(source)
    scaled_record.update(evaluated)
    scaled_record["design_id"] = scaled_id
    scaled_config = deepcopy(config)
    scaled_config["service_scenarios"] = [scenario]
    resolved = size_fixed_service(pd.DataFrame([scaled_record]), scaled_config).iloc[0].to_dict()
    bed = float(resolved["unit_bed_pressure_drop_kPa"])
    header = float(resolved["header_pressure_drop_kPa"])
    record = dict(resolved)
    record.update({
        "reference_design_id": source["design_id"],
        "linear_scale_factor": factor,
        "scaled_internal_diameter_m": geometry.internal_diameter_m,
        "scaled_bed_height_m": geometry.bed_height_m,
        "scaled_n_cells": scaled_n_cells,
        "scaled_nominal_unit_power_kW": float(evaluated["nominal_thermal_input_kW"]),
        "scaled_storage_volume_m3": float(evaluated["storage_volume_m3"]),
        "total_unit_plus_header_pressure_drop_kPa": bed + header,
        "bed_fraction_of_total_pressure_drop_pct": 100.0 * bed / max(bed + header, 1e-12),
        "header_to_bed_pressure_drop_ratio": header / max(bed, 1e-12),
        "extrapolative_scale_test": not math.isclose(factor, 1.0),
        "scale_interpretation": config["module_scale_sensitivity"]["interpretation"],
    })
    for ceiling in ceilings:
        record[f"within_{ceiling}_module_ceiling"] = int(resolved["parallel_unit_count"]) <= ceiling
    return record


def module_scale_sensitivity(
    unit: pd.DataFrame,
    decision_front: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Scale retained reference designs with deterministic outer parallelism."""
    unit_by_id = unit.set_index("design_id", drop=False)
    scenario_by_id = {row["scenario_id"]: row for row in config["service_scenarios"]}
    factors = [float(x) for x in config["module_scale_sensitivity"]["linear_scale_factors"]]
    ceilings = [int(x) for x in config["module_scale_sensitivity"]["module_count_ceilings"]]
    tasks = [
        (
            unit_by_id.loc[str(front_row["design_id"])].to_dict(),
            deepcopy(scenario_by_id[str(front_row["scenario_id"])]), factor, ceilings, config,
        )
        for front_row in decision_front.to_dict("records")
        for factor in factors
    ]
    workers = int(config.get("execution", {}).get("service_parallel_workers", 1))
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_module_scale_case, tasks))
    else:
        rows = [_module_scale_case(task) for task in tasks]
    return pd.DataFrame(rows)


def _transport_temperature_case(
    task: tuple[dict[str, Any], dict[str, Any], str, float, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one independent retained-design/transport-temperature case."""
    source, scenario, label, temperature, config = task
    scenario = deepcopy(scenario)
    scenario["transport_property_temperature_K"] = temperature
    local_config = deepcopy(config)
    local_config["service_scenarios"] = [scenario]
    result = size_fixed_service(pd.DataFrame([source]), local_config).iloc[0].to_dict()
    result.update({
        "reference_design_id": source["design_id"],
        "transport_temperature_case": label,
        "transport_temperature_sensitivity_interpretation": (
            "Deterministic endpoint test of run-wise frozen transport properties; capacity retains the registered temperature integral."
        ),
    })
    return result


def transport_temperature_sensitivity(
    unit: pd.DataFrame,
    decision_front: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Re-close decision-front cases with deterministic outer parallelism."""
    unit_by_id = unit.set_index("design_id", drop=False)
    scenario_by_id = {row["scenario_id"]: row for row in config["service_scenarios"]}
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    temperatures = [("cold_endpoint", cold_K), ("mean", 0.5 * (cold_K + hot_K)), ("hot_endpoint", hot_K)]
    tasks = [
        (
            unit_by_id.loc[str(front_row["design_id"])].to_dict(),
            scenario_by_id[str(front_row["scenario_id"])], label, temperature, config,
        )
        for front_row in decision_front.to_dict("records")
        for label, temperature in temperatures
    ]
    workers = int(config.get("execution", {}).get("service_parallel_workers", 1))
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_transport_temperature_case, tasks))
    else:
        rows = [_transport_temperature_case(task) for task in tasks]
    frame = pd.DataFrame(rows)
    keys = ["scenario_id", "reference_design_id"]
    mean = frame[frame["transport_temperature_case"].eq("mean")][
        keys + ["parallel_unit_count", "installed_packed_volume_m3", "unit_bed_pressure_drop_kPa"]
    ].rename(columns={
        "parallel_unit_count": "mean_transport_parallel_unit_count",
        "installed_packed_volume_m3": "mean_transport_installed_packed_volume_m3",
        "unit_bed_pressure_drop_kPa": "mean_transport_unit_bed_pressure_drop_kPa",
    })
    frame = frame.merge(mean, on=keys, how="left", validate="many_to_one")
    frame["parallel_unit_count_change_pct"] = 100.0 * (
        frame["parallel_unit_count"] / frame["mean_transport_parallel_unit_count"] - 1.0
    )
    frame["installed_volume_change_pct"] = 100.0 * (
        frame["installed_packed_volume_m3"] / frame["mean_transport_installed_packed_volume_m3"] - 1.0
    )
    frame["bed_pressure_drop_change_pct"] = 100.0 * (
        frame["unit_bed_pressure_drop_kPa"] / frame["mean_transport_unit_bed_pressure_drop_kPa"] - 1.0
    )
    return frame


def _service_capacity_scan(
    source: dict[str, Any],
    scenario: dict[str, Any],
    config: dict[str, Any],
    counts: Iterable[int],
) -> pd.DataFrame:
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    dT = hot_K - cold_K
    cp = float(source.get("mean_cp_J_kgK", fluid_properties(str(source["fluid_model_id"]), 0.5 * (cold_K + hot_K)).cp_J_kgK))
    power_MW = float(scenario["target_thermal_power_MW"])
    required_energy = power_MW * float(scenario["storage_duration_h"])
    total_flow = power_MW * 1e6 / (cp * dT)
    geometry = BedGeometryV16(
        internal_diameter_m=float(source["internal_diameter_m"]),
        bed_height_m=float(source["bed_height_m"]),
        porosity=float(source["porosity"]),
        particle_diameter_m=float(source["particle_diameter_m"]),
        topology=str(source["topology"]),
        gradient_ratio=float(source["gradient_ratio"]),
        gradient_orientation=str(source["gradient_orientation"]),
        layer_fraction=float(source["layer_fraction"]),
    )
    solid = config["reference_geometry"]
    multipliers = {
        "density": float(source.get("density_multiplier", 1.0)),
        "cp": float(source.get("cp_multiplier", 1.0)),
        "viscosity": float(source.get("viscosity_multiplier", 1.0)),
        "conductivity": float(source.get("conductivity_multiplier", 1.0)),
    }
    rows: list[dict[str, Any]] = []
    for count in counts:
        n = int(count)
        actual = evaluate_packed_bed_v16(
            model_id=str(source["fluid_model_id"]),
            design_id=f"{source['design_id']}__{scenario['scenario_id']}__MONO_N{n}",
            geometry=geometry,
            mass_flow_kg_s=total_flow / n,
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
        )
        capacity = n * float(actual["deliverable_energy_MWh"])
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "reference_design_id": source["design_id"],
            "parallel_unit_count": n,
            "installed_deliverable_energy_MWh": capacity,
            "required_energy_MWh": required_energy,
            "capacity_feasible": capacity + 1e-12 >= required_energy,
            "resolved_unit_mass_flow_kg_s": total_flow / n,
            "resolved_unit_deliverable_energy_MWh": actual["deliverable_energy_MWh"],
        })
    return pd.DataFrame(rows)


def _service_monotonicity_audit_case(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int],
) -> pd.DataFrame:
    """Exhaustively audit one retained decision without nested parallelism."""
    source, scenario, config, lower, upper = task
    frame = _service_capacity_scan(source, scenario, config, range(lower, upper + 1))
    delta = frame["installed_deliverable_energy_MWh"].diff()
    frame["capacity_increment_MWh"] = delta
    frame["nondecreasing_from_previous_count"] = delta.isna() | (delta >= -1e-10)
    feasible_counts = frame.loc[frame["capacity_feasible"], "parallel_unit_count"]
    first_feasible = None if feasible_counts.empty else int(feasible_counts.min())
    frame["reported_minimum_unit_count"] = upper
    frame["first_scanned_feasible_unit_count"] = first_feasible
    frame["reported_minimum_matches_full_scan"] = first_feasible == upper
    frame["full_integer_scan"] = True
    return frame


def service_monotonicity_audit(
    unit: pd.DataFrame,
    decision_front: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Exhaustively scan retained integer counts with deterministic outer parallelism."""
    unit_by_id = unit.set_index("design_id", drop=False)
    scenario_by_id = {row["scenario_id"]: row for row in config["service_scenarios"]}
    maximum_span = int(config["service_monotonicity_audit"]["maximum_full_integer_scan_span"])
    tasks: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]] = []
    for front_row in decision_front.to_dict("records"):
        lower = int(front_row["power_limited_unit_count"])
        upper = int(front_row["parallel_unit_count"])
        if upper - lower > maximum_span:
            raise ValueError(f"registered monotonicity scan span exceeded for {front_row['design_id']}")
        tasks.append((
            unit_by_id.loc[str(front_row["design_id"])].to_dict(),
            scenario_by_id[str(front_row["scenario_id"])], config, lower, upper,
        ))
    workers = int(config.get("execution", {}).get("audit_parallel_workers", 1))
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            frames = list(executor.map(_service_monotonicity_audit_case, tasks))
    else:
        frames = [_service_monotonicity_audit_case(task) for task in tasks]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def service_monotonicity_probe(
    unit: pd.DataFrame,
    service: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Search a stratified service sample for non-monotone capacity behavior.

    The binary bracket is followed by an exhaustive scan for every retained
    decision case.  This broader probe is not advertised as a proof for every
    rejected design; it is a fail-closed counterexample search across fluid,
    topology, service duty and count regimes.
    """
    source_by_id = unit.set_index("design_id", drop=False)
    scenario_by_id = {row["scenario_id"]: row for row in config["service_scenarios"]}
    selections: list[pd.Series] = []
    eligible = service[service["service_admissible"]].copy()
    for _, part in eligible.groupby(["scenario_id", "fluid_model_id", "topology_id"], sort=True):
        ordered = part.sort_values("parallel_unit_count").reset_index(drop=True)
        positions = sorted(set(int(round(x * (len(ordered) - 1))) for x in (0.10, 0.50, 0.90)))
        selections.extend(ordered.iloc[positions].itertuples(index=False, name=None))
    rows: list[pd.DataFrame] = []
    columns = list(eligible.columns)
    for values in selections:
        selected = dict(zip(columns, values))
        lower = int(selected["power_limited_unit_count"])
        upper = int(selected["parallel_unit_count"])
        if upper <= lower:
            counts = [lower]
        else:
            counts = np.unique(np.rint(np.geomspace(lower, upper, num=9)).astype(int)).tolist()
            counts = sorted(set([lower, *counts, upper]))
        frame = _service_capacity_scan(
            source_by_id.loc[str(selected["design_id"])].to_dict(),
            scenario_by_id[str(selected["scenario_id"])],
            config,
            counts,
        )
        frame["probe_kind"] = "stratified_log_count_counterexample_search"
        frame["reported_minimum_unit_count"] = upper
        delta = frame["installed_deliverable_energy_MWh"].diff()
        frame["capacity_increment_MWh"] = delta
        frame["nondecreasing_from_previous_count"] = delta.isna() | (delta >= -1e-10)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def decision_grid_refinement(
    unit: pd.DataFrame,
    decision_front: pd.DataFrame,
    config: dict[str, Any],
    *,
    enforce: bool = True,
) -> pd.DataFrame:
    """Reclose all retained decisions on the registered refinement mesh.

    Selection remains a 2048-cell epsilon archive.  The independent 4096-cell
    reclosure is a release gate for the numerical reporting resolution, rather
    than a post-hoc source of more favorable objective values.
    """
    spec = config["numerical_decision_validation"]
    production_cells = int(spec["production_n_cells"])
    refinement_cells = int(spec["refinement_n_cells"])
    if set(unit["n_cells"].astype(int)) != {production_cells}:
        raise ValueError("grid-refinement base must use the registered production mesh")
    if refinement_cells <= production_cells:
        raise ValueError("refinement mesh must be finer than production mesh")
    keys = ["scenario_id", "design_id"]
    requested = decision_front[keys].drop_duplicates().copy()
    source = unit.merge(requested[["design_id"]].drop_duplicates(), on="design_id", how="inner")
    source = source.copy()
    source["n_cells"] = refinement_cells
    refined = size_fixed_service(source, config)
    baseline_columns = keys + [
        "parallel_unit_count", "resolved_unit_deliverable_energy_MWh",
        "resolved_high_grade_discharge_fraction", "installed_packed_volume_m3",
        "salt_inventory_t", "pump_energy_fraction", "capacity_oversize_pct",
    ]
    base = decision_front[baseline_columns].rename(columns={
        "parallel_unit_count": "production_parallel_unit_count",
        "resolved_unit_deliverable_energy_MWh": "production_resolved_unit_deliverable_energy_MWh",
        "resolved_high_grade_discharge_fraction": "production_resolved_high_grade_discharge_fraction",
        "installed_packed_volume_m3": "production_installed_packed_volume_m3",
        "salt_inventory_t": "production_salt_inventory_t",
        "pump_energy_fraction": "production_pump_energy_fraction",
        "capacity_oversize_pct": "production_capacity_oversize_pct",
    })
    refined = refined.merge(base, on=keys, how="inner", validate="one_to_one")
    refined["production_n_cells"] = production_cells
    refined["refinement_n_cells"] = refinement_cells
    refined["deliverable_energy_change_pct"] = 100.0 * (
        refined["resolved_unit_deliverable_energy_MWh"]
        / refined["production_resolved_unit_deliverable_energy_MWh"] - 1.0
    )
    refined["high_grade_fraction_change"] = (
        refined["resolved_high_grade_discharge_fraction"]
        - refined["production_resolved_high_grade_discharge_fraction"]
    )
    refined["parallel_unit_count_change_pct"] = 100.0 * (
        refined["parallel_unit_count"] / refined["production_parallel_unit_count"] - 1.0
    )
    refined["grid_refinement_accepts_deliverable_energy"] = (
        refined["deliverable_energy_change_pct"].abs()
        <= float(spec["maximum_absolute_deliverable_energy_change_pct"])
    )
    refined["grid_refinement_accepts_high_grade_fraction"] = (
        refined["high_grade_fraction_change"].abs()
        <= float(spec["maximum_absolute_high_grade_fraction_change"])
    )
    refined["grid_refinement_accepts_parallel_unit_count"] = (
        refined["parallel_unit_count_change_pct"].abs()
        <= float(spec["maximum_absolute_parallel_unit_count_change_pct"])
    )
    refined["grid_refinement_accepted"] = (
        refined["grid_refinement_accepts_deliverable_energy"]
        & refined["grid_refinement_accepts_high_grade_fraction"]
        & refined["grid_refinement_accepts_parallel_unit_count"]
        & refined["service_admissible"]
    )
    refined["grid_refinement_scope"] = str(spec["scope"])
    if enforce and not bool(refined["grid_refinement_accepted"].all()):
        bad = refined.loc[~refined["grid_refinement_accepted"], keys].to_dict("records")
        raise RuntimeError(f"V16 grid-refinement release gate failed: {bad}")
    return refined


def exact_service_pareto(service: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fronts = []
    for scenario, part in service[service["service_admissible"]].groupby("scenario_id"):
        part = part.reset_index(drop=True)
        front = part.loc[pareto_mask(part, config["service_objectives"])].copy()
        front["pareto_scope"] = "exact_fixed_service_audit_front"
        fronts.append(front)
    return pd.concat(fronts, ignore_index=True) if fronts else pd.DataFrame()


def _decision_widths(part: pd.DataFrame, config: dict[str, Any], multiplier: float = 1.0) -> dict[str, float]:
    precision = config["decision_precision"]
    widths: dict[str, float] = {}
    for key, _ in config["service_objectives"]:
        if key == "capacity_oversize_pct":
            width = float(precision["capacity_oversize_absolute_pct"])
        else:
            rel = float(precision["relative_fraction"][key])
            scale = max(abs(float(part[key].median())), abs(float(part[key].min())), 1e-12)
            width = rel * scale
        widths[key] = max(width * multiplier, 1e-12)
    return widths


def decision_resolution_service_pareto(
    service: pd.DataFrame,
    config: dict[str, Any],
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """Build a deterministic epsilon-box archive at declared decision precision."""
    if multiplier <= 0:
        return exact_service_pareto(service, config)
    fronts: list[pd.DataFrame] = []
    objective_keys = [x[0] for x in config["service_objectives"]]
    for scenario, part in service[service["service_admissible"]].groupby("scenario_id"):
        part = part.reset_index(drop=True).copy()
        widths = _decision_widths(part, config, multiplier)
        box_cols: list[str] = []
        centre_distance = np.zeros(len(part), dtype=float)
        for key in objective_keys:
            origin = float(part[key].min())
            coord = np.floor((part[key].to_numpy(float) - origin) / widths[key] + 1e-12).astype(int)
            col = f"epsilon_box_{key}"
            part[col] = coord
            box_cols.append(col)
            position = (part[key].to_numpy(float) - origin) / widths[key] - coord
            centre_distance += (position - 0.5) ** 2
        part["epsilon_box_centre_distance"] = np.sqrt(centre_distance)
        representatives = (
            part.sort_values(["epsilon_box_centre_distance", "design_id"])
            .drop_duplicates(box_cols, keep="first")
            .reset_index(drop=True)
        )
        box_objectives = [[col, "minimize"] for col in box_cols]
        front = representatives.loc[pareto_mask(representatives, box_objectives)].copy()
        front["pareto_scope"] = "decision_resolution_fixed_service_front"
        front["decision_precision_multiplier"] = float(multiplier)
        for key, width in widths.items():
            front[f"decision_width_{key}"] = width
        fronts.append(front)
    return pd.concat(fronts, ignore_index=True) if fronts else pd.DataFrame()


def pareto_precision_sensitivity(service: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for multiplier in config["decision_precision"]["sensitivity_multipliers"]:
        front = (
            exact_service_pareto(service, config)
            if float(multiplier) == 0.0
            else decision_resolution_service_pareto(service, config, float(multiplier))
        )
        for scenario, part in front.groupby("scenario_id"):
            rows.append({
                "scenario_id": scenario,
                "decision_precision_multiplier": float(multiplier),
                "front_definition": "exact" if float(multiplier) == 0.0 else "epsilon_box",
                "n_front_designs": int(len(part)),
                "n_fluids": int(part["fluid_model_id"].nunique()),
                "n_topologies": int(part["topology_id"].nunique()),
            })
    return pd.DataFrame(rows)


def advantage_transfer(results: pd.DataFrame, service: pd.DataFrame) -> pd.DataFrame:
    unit = results.copy()
    unit["unit_high_grade_energy_density_MWh_m3"] = unit["stored_heat_MWh"] * unit["high_grade_discharge_fraction"] / unit["storage_volume_m3"]
    keys = ["base_design_id", "topology_id"]
    reference_id = "SOLAR_SALT_60_40_V10"
    ref_unit = unit[unit["fluid_model_id"] == reference_id][keys + [
        "fluid_volumetric_sensible_kWh_m3", "volumetric_capacity_MWh_m3", "unit_high_grade_energy_density_MWh_m3"
    ]].rename(columns={
        "fluid_volumetric_sensible_kWh_m3": "ref_material",
        "volumetric_capacity_MWh_m3": "ref_unit",
        "unit_high_grade_energy_density_MWh_m3": "ref_dynamic",
    })
    merged = unit.merge(ref_unit, on=keys, how="inner")
    merged = merged[merged["fluid_model_id"] != reference_id].copy()
    merged["A_material_log"] = np.log(merged["fluid_volumetric_sensible_kWh_m3"] / merged["ref_material"])
    merged["A_unit_log"] = np.log(merged["volumetric_capacity_MWh_m3"] / merged["ref_unit"])
    merged["A_dynamic_log"] = np.log(merged["unit_high_grade_energy_density_MWh_m3"] / merged["ref_dynamic"])
    svc_ref = service[service["fluid_model_id"] == reference_id][["scenario_id"] + keys + ["service_compactness_MWh_m3"]].rename(columns={"service_compactness_MWh_m3": "ref_system"})
    svc = service.merge(svc_ref, on=["scenario_id"] + keys, how="inner")
    svc = svc[svc["fluid_model_id"] != reference_id][["scenario_id", "fluid_model_id"] + keys + ["service_compactness_MWh_m3", "ref_system"]]
    svc["A_system_log"] = np.log(svc["service_compactness_MWh_m3"] / svc["ref_system"])
    out = merged.merge(svc, on=["fluid_model_id"] + keys, how="inner")
    def ratio(num: pd.Series, den: pd.Series) -> np.ndarray:
        return np.where(np.abs(den) > 1e-10, num / den, np.nan)
    out["ARI_material_to_unit"] = ratio(out["A_unit_log"], out["A_material_log"])
    out["ARI_unit_to_dynamic"] = ratio(out["A_dynamic_log"], out["A_unit_log"])
    out["ARI_dynamic_to_system"] = ratio(out["A_system_log"], out["A_dynamic_log"])
    out["ARI_material_to_system"] = ratio(out["A_system_log"], out["A_material_log"])
    conditions = [out["A_system_log"] * out["A_material_log"] < 0, np.abs(out["A_system_log"]) > np.abs(out["A_material_log"]), np.abs(out["A_system_log"]) < np.abs(out["A_material_log"])]
    choices = ["reversal", "amplification", "contraction"]
    out["advantage_fate"] = np.select(conditions, choices, default="retained")
    cols = [
        "scenario_id", "fluid_model_id", "fluid_name", "base_design_id", "topology_id", "topology", "gradient_orientation",
        "A_material_log", "A_unit_log", "A_dynamic_log", "A_system_log",
        "ARI_material_to_unit", "ARI_unit_to_dynamic", "ARI_dynamic_to_system", "ARI_material_to_system", "advantage_fate",
    ]
    return out[cols]


def pairwise_scale_order(results: pd.DataFrame, service: pd.DataFrame) -> pd.DataFrame:
    """Audit every matched fluid pair for reversal or exact service-scale ties."""
    keys = ["scenario_id", "base_design_id", "topology_id"]
    material = results[[
        "base_design_id", "topology_id", "fluid_model_id", "fluid_name",
        "fluid_volumetric_sensible_kWh_m3",
    ]].copy()
    matched = service[keys + [
        "fluid_model_id", "fluid_name", "service_compactness_MWh_m3",
    ]].merge(
        material,
        on=["base_design_id", "topology_id", "fluid_model_id", "fluid_name"],
        how="left",
        validate="many_to_one",
    )
    left = matched.rename(columns={
        "fluid_model_id": "fluid_model_id_i",
        "fluid_name": "fluid_name_i",
        "fluid_volumetric_sensible_kWh_m3": "material_capacity_i_kWh_m3",
        "service_compactness_MWh_m3": "system_compactness_i_MWh_m3",
    })
    right = matched.rename(columns={
        "fluid_model_id": "fluid_model_id_j",
        "fluid_name": "fluid_name_j",
        "fluid_volumetric_sensible_kWh_m3": "material_capacity_j_kWh_m3",
        "service_compactness_MWh_m3": "system_compactness_j_MWh_m3",
    })
    pairs = left.merge(right, on=keys, how="inner")
    pairs = pairs[pairs["fluid_model_id_i"] < pairs["fluid_model_id_j"]].copy()
    pairs["material_log_difference"] = np.log(
        pairs["material_capacity_i_kWh_m3"] / pairs["material_capacity_j_kWh_m3"]
    )
    pairs["system_log_difference"] = np.log(
        pairs["system_compactness_i_MWh_m3"] / pairs["system_compactness_j_MWh_m3"]
    )
    pairs["system_exact_tie"] = (
        pairs["system_compactness_i_MWh_m3"] == pairs["system_compactness_j_MWh_m3"]
    )
    pairs["pair_fate"] = np.select(
        [
            pairs["material_log_difference"] * pairs["system_log_difference"] < 0,
            pairs["system_exact_tie"] & (pairs["material_log_difference"] != 0),
        ],
        ["reversal", "system_exact_tie"],
        default="order_preserved",
    )
    return pairs[[
        *keys, "fluid_model_id_i", "fluid_name_i", "fluid_model_id_j", "fluid_name_j",
        "material_capacity_i_kWh_m3", "material_capacity_j_kWh_m3",
        "system_compactness_i_MWh_m3", "system_compactness_j_MWh_m3",
        "material_log_difference", "system_log_difference", "system_exact_tie", "pair_fate",
    ]].sort_values(keys + ["fluid_model_id_i", "fluid_model_id_j"]).reset_index(drop=True)


def interaction_indices(results: pd.DataFrame) -> pd.DataFrame:
    variables = ["internal_diameter_m", "bed_height_m", "porosity", "particle_diameter_m", "nominal_thermal_input_kW"]
    metrics = ["volumetric_capacity_MWh_m3", "charge_t90_h", "pressure_drop_kPa", "high_grade_discharge_fraction"]
    rows: list[dict[str, Any]] = []
    for fluid, part in results[results["physics_comparison_admissible"]].groupby("fluid_model_id"):
        x0 = part[variables].to_numpy(float)
        x0 = (x0 - x0.mean(axis=0)) / x0.std(axis=0)
        topo_fine = ((part["topology"] == "graded_two_layer") & (part["gradient_orientation"] == "fine_hot")).to_numpy(float)
        topo_coarse = ((part["topology"] == "graded_two_layer") & (part["gradient_orientation"] == "coarse_hot")).to_numpy(float)
        xbase = np.column_stack([x0, topo_fine, topo_coarse])
        names = variables + ["graded_fine_hot", "graded_coarse_hot"]
        interaction_terms = []
        interaction_names = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                interaction_terms.append(xbase[:, i] * xbase[:, j])
                interaction_names.append(f"{names[i]} × {names[j]}")
        X = np.column_stack([np.ones(len(part)), xbase] + interaction_terms)
        for metric in metrics:
            y = part[metric].to_numpy(float)
            y = (y - y.mean()) / max(y.std(), 1e-12)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ coef
            sse_full = float(np.sum((y - pred) ** 2))
            sst = float(np.sum((y - y.mean()) ** 2))
            for k, name in enumerate(interaction_names):
                col = 1 + len(names) + k
                Xr = np.delete(X, col, axis=1)
                cr = np.linalg.lstsq(Xr, y, rcond=None)[0]
                sse_reduced = float(np.sum((y - Xr @ cr) ** 2))
                rows.append({
                    "fluid_model_id": fluid,
                    "metric": metric,
                    "interaction": name,
                    "standardized_coefficient": float(coef[col]),
                    "partial_R2": max(0.0, (sse_reduced - sse_full) / max(sst, 1e-12)),
                    "full_model_R2": 1.0 - sse_full / max(sst, 1e-12),
                    "n_cases": len(part),
                })
    return pd.DataFrame(rows).sort_values(["fluid_model_id", "metric", "partial_R2"], ascending=[True, True, False])


def topology_response_study(config_path: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    """Resolve gradient ratio and split location on matched unit geometries."""
    spec = config["topology_response"]
    base_expanded = generate_geometry_pool(config, int(spec["matched_base_geometries"]), int(spec["seed"]))
    bases = base_expanded.sort_values("topology_id").drop_duplicates("base_design_id")
    rows: list[dict[str, Any]] = []
    for base in bases.to_dict("records"):
        homogeneous = dict(base)
        homogeneous.update({
            "topology": "homogeneous", "gradient_ratio": 1.0,
            "gradient_orientation": "none", "layer_fraction": 0.5,
            "topology_id": "TR_H", "design_id": f"{base['base_design_id']}_TR_H",
        })
        rows.append(homogeneous)
        for ratio in spec["gradient_ratios"]:
            if float(ratio) <= 1.0:
                continue
            for fraction in spec["layer_fractions"]:
                for orientation in spec["orientations"]:
                    graded = dict(base)
                    graded.update({
                        "topology": "graded_two_layer",
                        "gradient_ratio": float(ratio),
                        "gradient_orientation": str(orientation),
                        "layer_fraction": float(fraction),
                        "topology_id": f"TR_G{float(ratio):.1f}_F{float(fraction):.1f}_{orientation}",
                        "design_id": f"{base['base_design_id']}_TR_G{float(ratio):.1f}_F{float(fraction):.1f}_{orientation}",
                    })
                    rows.append(graded)
    pool = pd.DataFrame(rows)
    evaluated = evaluate_geometry_pool(config_path, config, pool, int(spec["n_cells"]))
    keys = ["base_design_id", "fluid_model_id"]
    reference = evaluated[evaluated["topology"] == "homogeneous"][keys + [
        "high_grade_discharge_fraction", "pressure_drop_kPa", "charge_t90_h",
        "thermal_front_thickness_ratio",
    ]].rename(columns={
        "high_grade_discharge_fraction": "reference_high_grade_discharge_fraction",
        "pressure_drop_kPa": "reference_pressure_drop_kPa",
        "charge_t90_h": "reference_charge_t90_h",
        "thermal_front_thickness_ratio": "reference_thermal_front_thickness_ratio",
    })
    out = evaluated.merge(reference, on=keys, how="left")
    out["high_grade_change_percentage_point"] = 100.0 * (
        out["high_grade_discharge_fraction"] - out["reference_high_grade_discharge_fraction"]
    )
    for metric in ("pressure_drop_kPa", "charge_t90_h", "thermal_front_thickness_ratio"):
        out[f"{metric}_change_pct"] = 100.0 * (
            out[metric] / out[f"reference_{metric}"] - 1.0
        )
    return out


def _normalise_min_objectives(frame: pd.DataFrame, objective_keys: list[str], reference: pd.DataFrame | None = None) -> np.ndarray:
    ref = frame if reference is None else reference
    out = np.zeros((len(frame), len(objective_keys)))
    for j, key in enumerate(objective_keys):
        lo = float(ref[key].min()); hi = float(ref[key].max())
        out[:, j] = 0.0 if hi == lo else (frame[key].to_numpy(float) - lo) / (hi - lo)
    return out


def _igd_plus(front: pd.DataFrame, reference: pd.DataFrame, keys: list[str]) -> float:
    if front.empty or reference.empty:
        return float("nan")
    a = _normalise_min_objectives(front, keys, reference)
    b = _normalise_min_objectives(reference, keys, reference)
    distances = []
    for point in b:
        diff = np.maximum(a - point, 0.0)
        distances.append(np.min(np.linalg.norm(diff, axis=1)))
    return float(np.mean(distances))


def _epsilon_indicator(front: pd.DataFrame, reference: pd.DataFrame, keys: list[str]) -> float:
    if front.empty or reference.empty:
        return float("nan")
    a = _normalise_min_objectives(front, keys, reference)
    b = _normalise_min_objectives(reference, keys, reference)
    eps_values = []
    for point in b:
        eps_values.append(np.min(np.max(a - point, axis=1)))
    return float(np.max(eps_values))


def sampling_robustness(config_path: str | Path, config: dict[str, Any]) -> pd.DataFrame:
    runs: list[dict[str, Any]] = []
    fronts: dict[tuple[int, int, str], pd.DataFrame] = {}
    keys = [x[0] for x in config["service_objectives"]]
    for n in config["sampling"]["robustness_sample_sizes"]:
        for seed in config["sampling"]["robustness_seeds"]:
            pool = generate_geometry_pool(config, int(n), int(seed))
            unit = evaluate_geometry_pool(config_path, config, pool, int(config["sampling"]["robustness_n_cells"]))
            service = size_fixed_service(unit, config)
            for scenario_id in [x["scenario_id"] for x in config["service_scenarios"]]:
                part = service[(service["scenario_id"] == scenario_id) & service["service_admissible"]].reset_index(drop=True)
                front = decision_resolution_service_pareto(part, config)
                fronts[(int(n), int(seed), str(scenario_id))] = front
                runs.append({
                    "scenario_id": scenario_id,
                    "n_base_geometries": int(n), "seed": int(seed), "n_unit_cases": len(unit),
                    "n_admissible_unit": int(unit["physics_comparison_admissible"].sum()),
                    "n_service_cases": len(part), "n_service_pareto": len(front),
                    "objective_definition": ";".join(keys),
                })
    max_n = max(config["sampling"]["robustness_sample_sizes"])
    for scenario_id in [x["scenario_id"] for x in config["service_scenarios"]]:
        pooled = pd.concat(
            [fronts[(max_n, int(s), str(scenario_id))] for s in config["sampling"]["robustness_seeds"]],
            ignore_index=True,
        )
        reference = pooled.loc[pareto_mask(pooled, [[k, "minimize"] for k in keys])].copy()
        for row in runs:
            if row["scenario_id"] != scenario_id:
                continue
            f = fronts[(row["n_base_geometries"], row["seed"], str(scenario_id))]
            row["IGD_plus"] = _igd_plus(f, reference, keys)
            row["additive_epsilon_indicator"] = _epsilon_indicator(f, reference, keys)
    return pd.DataFrame(runs)


def bounded_robustness(
    config_path: str | Path,
    config: dict[str, Any],
    pool: pd.DataFrame,
    nominal_unit: pd.DataFrame,
    nominal_service: pd.DataFrame,
) -> pd.DataFrame:
    spec = config["bounded_property_uncertainty"]
    scenario_ids = [x["scenario_id"] for x in config["service_scenarios"]]
    stable_nominal = decision_resolution_service_pareto(nominal_service, config)
    selected_ids = sorted(set(stable_nominal["design_id"]))
    selected_pool_design_ids = {did.split("__", 1)[0] for did in selected_ids}
    candidate_pool = pool[pool["design_id"].isin(selected_pool_design_ids)].copy()
    records: list[dict[str, Any]] = []
    for model_id, bounds in spec["models"].items():
        model_nominal_ids = [x for x in selected_ids if x.endswith("__" + model_id)]
        if not model_nominal_ids:
            continue
        properties = list(bounds)
        samples = _lhs(int(spec["stress_samples"]), len(properties), int(spec["seed"]) + list(spec["models"]).index(model_id))
        front_count = {(scenario, did): 0 for scenario in scenario_ids for did in model_nominal_ids}
        scores: dict[tuple[str, str], list[float]] = {(scenario, did): [] for scenario in scenario_ids for did in model_nominal_ids}
        for s in range(int(spec["stress_samples"])):
            mult = {}
            for j, prop in enumerate(properties):
                lo, hi = map(float, bounds[prop])
                mult[prop] = lo + samples[s, j] * (hi - lo)
            unit = evaluate_geometry_pool(
                config_path, config, candidate_pool, int(config["sampling"]["robustness_n_cells"]),
                {model_id: mult}, model_ids=[model_id]
            )
            svc = size_fixed_service(unit, config)
            for scenario_id in scenario_ids:
                part = svc[(svc["scenario_id"] == scenario_id) & svc["design_id"].isin(model_nominal_ids) & svc["service_admissible"]].reset_index(drop=True)
                if part.empty:
                    continue
                keys = [x[0] for x in config["service_objectives"]]
                front = decision_resolution_service_pareto(part, config)
                for did in front["design_id"]:
                    front_count[(scenario_id, did)] += 1
                norm = _normalise_min_objectives(part, keys)
                score = np.sqrt(np.mean(norm**2, axis=1))
                best = float(np.min(score))
                for did, sc in zip(part["design_id"], score):
                    scores[(scenario_id, did)].append(100.0 * (float(sc) - best) / max(best, 1e-9))
        for scenario_id in scenario_ids:
            for did in model_nominal_ids:
                arr = np.asarray(scores[(scenario_id, did)], dtype=float)
                records.append({
                    "scenario_id": scenario_id,
                    "fluid_model_id": model_id,
                    "design_id": did,
                    "registered_bound_properties": ";".join(properties),
                    "n_stress_samples": int(spec["stress_samples"]),
                    "pareto_attainment_frequency_pct": 100.0 * front_count[(scenario_id, did)] / int(spec["stress_samples"]),
                    "median_equal_weight_regret_pct": float(np.nanmedian(arr)) if arr.size else np.nan,
                    "max_equal_weight_regret_pct": float(np.nanmax(arr)) if arr.size else np.nan,
                    "interpretation": "non-probabilistic Latin-hypercube stress frequency under registered bounds",
                })
    # Preserve missing uncertainty as an explicit gate.
    bounded_models = set(spec["models"])
    for scenario_id in scenario_ids:
        for model_id in config["fluid_model_ids"]:
            if model_id not in bounded_models:
                records.append({
                    "scenario_id": scenario_id,
                    "fluid_model_id": model_id,
                    "design_id": "NOT_ASSESSED",
                    "registered_bound_properties": "none_registered",
                    "n_stress_samples": 0,
                    "pareto_attainment_frequency_pct": np.nan,
                    "median_equal_weight_regret_pct": np.nan,
                    "max_equal_weight_regret_pct": np.nan,
                    "interpretation": "uncertainty not quantified in registered source; no distribution or stability claim invented",
                })
    return pd.DataFrame(records)


def model_form_stress(
    config_path: str | Path,
    config: dict[str, Any],
    pool: pd.DataFrame,
    nominal_service: pd.DataFrame,
) -> pd.DataFrame:
    """Re-evaluate the nominal decision archive under deterministic physics stresses."""
    stable = decision_resolution_service_pareto(nominal_service, config)
    selected_ids = set(stable["design_id"])
    selected_pool_ids = {x.split("__", 1)[0] for x in selected_ids}
    candidate_pool = pool[pool["design_id"].isin(selected_pool_ids)].copy()
    records: list[pd.DataFrame] = []
    for stress in config["model_form_stress"]["scenarios"]:
        overrides = {
            "axial_dispersion_m2_s": float(stress["axial_dispersion_m2_s"]),
            "solid_axial_diffusivity_m2_s": float(stress["solid_axial_diffusivity_m2_s"]),
            "surface_heat_transfer_coefficient_W_m2K": float(stress["surface_heat_transfer_coefficient_W_m2K"]),
            "ambient_temperature_C": float(config["model_form_stress"]["ambient_temperature_C"]),
        }
        unit = evaluate_geometry_pool(
            config_path, config, candidate_pool,
            int(config["model_form_stress"]["n_cells"]),
            physics_overrides=overrides,
        )
        unit = unit[unit["design_id"].isin(selected_ids)].copy()
        service = size_fixed_service(unit, config)
        front = decision_resolution_service_pareto(service, config)
        member = set() if front.empty else set(zip(front["scenario_id"], front["design_id"]))
        service["stress_id"] = stress["stress_id"]
        # Carry every registered stress input into the result table.  This is
        # reporting metadata only: the numerical fields above were already
        # evaluated with the same overrides in ``evaluate_geometry_pool``.
        service["stress_ambient_temperature_C"] = overrides["ambient_temperature_C"]
        service["stress_axial_dispersion_m2_s"] = overrides["axial_dispersion_m2_s"]
        service["stress_solid_axial_diffusivity_m2_s"] = overrides["solid_axial_diffusivity_m2_s"]
        service["stress_surface_heat_transfer_coefficient_W_m2K"] = overrides[
            "surface_heat_transfer_coefficient_W_m2K"
        ]
        service["stress_n_cells"] = int(config["model_form_stress"]["n_cells"])
        service["decision_front_member"] = [
            (scenario, design) in member
            for scenario, design in zip(service["scenario_id"], service["design_id"])
        ]
        service["stress_interpretation"] = config["model_form_stress"]["interpretation"]
        records.append(service)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def standby_loss_screen(
    config: dict[str, Any],
    nominal_unit: pd.DataFrame,
    nominal_service: pd.DataFrame,
) -> pd.DataFrame:
    """Bound standby retention for designs in the nominal decision archive."""
    spec = config["standby_loss"]
    stable = decision_resolution_service_pareto(nominal_service, config)
    selected_ids = sorted(set(stable["design_id"]))
    unit = nominal_unit[nominal_unit["design_id"].isin(selected_ids)].copy()
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    ambient_K = float(spec["ambient_temperature_C"]) + 273.15
    delta_t = hot_K - cold_K
    rows: list[dict[str, Any]] = []
    for r in unit.to_dict("records"):
        cvol = float(r["stored_heat_MWh"]) * 3.6e9 / max(float(r["storage_volume_m3"]) * delta_t, 1e-12)
        area_to_volume = 4.0 / float(r["internal_diameter_m"]) + 2.0 / float(r["bed_height_m"])
        for surface_u in spec["surface_heat_transfer_coefficient_W_m2K"]:
            loss = float(surface_u) * area_to_volume
            for duration in spec["durations_h"]:
                mean_K = ambient_K + (hot_K - ambient_K) * math.exp(
                    -float(loss) * float(duration) * 3600.0 / max(cvol, 1e-12)
                )
                retained = min(1.0, max(0.0, (mean_K - cold_K) / delta_t))
                rows.append({
                    "design_id": r["design_id"],
                    "fluid_model_id": r["fluid_model_id"],
                    "fluid_name": r["fluid_name"],
                    "topology_id": r["topology_id"],
                    "standby_duration_h": float(duration),
                    "surface_heat_transfer_coefficient_W_m2K": float(surface_u),
                    "external_area_to_packed_volume_ratio_m1": area_to_volume,
                    "wall_loss_W_m3K": float(loss),
                    "ambient_temperature_C": float(spec["ambient_temperature_C"]),
                    "effective_volumetric_heat_capacity_J_m3K": cvol,
                    "mean_temperature_after_standby_C": mean_K - 273.15,
                    "retained_sensible_inventory_fraction": retained,
                    "standby_sensible_loss_pct": 100.0 * (1.0 - retained),
                    "interpretation": spec["interpretation"],
                })
    return pd.DataFrame(rows)


def reference_control_volume_closure(reference_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    geom = reference["storage_unit"]["reference_geometry"]
    bench = reference["storage_unit"]["benchmark_capacity"]
    solid = config["reference_geometry"]
    area = math.pi * float(geom["internal_diameter_m"]) ** 2 / 4.0
    volume = area * float(geom["packed_bed_height_m"])
    eps = float(geom["skeleton_porosity"])
    dT = float(bench["temperature_step_K"])
    filler = (1.0-eps)*volume*float(solid["solid_density_kg_m3"])*float(solid["solid_cp_J_kgK"])*dT/3.6e9
    T_C = 340.0
    density = 2090.0 - 0.636*T_C
    cp = 1443.0 + 0.172*T_C
    pore = eps*volume*density*cp*dT/3.6e9
    return {
        "reference_storage_volume_m3": volume,
        "derived_filler_MWh": filler,
        "reported_filler_MWh": float(bench["filler_MWh"]),
        "filler_relative_error_pct": 100.0*(filler/float(bench["filler_MWh"])-1.0),
        "derived_geometric_pore_salt_MWh": pore,
        "reported_salt_inventory_MWh": float(bench["salt_MWh"]),
        "geometric_pore_fraction_of_reported_salt_pct": 100.0*pore/float(bench["salt_MWh"]),
        "unresolved_reported_salt_inventory_MWh": float(bench["salt_MWh"])-pore,
        "interpretation": "Filler closes at component level. Salt reported by the source exceeds geometric pore inventory and remains an explicitly unresolved control-volume inventory rather than being forced into model error.",
    }


def _atomic_pickle_dump(payload: dict[str, Any], path: Path) -> None:
    """Write a restart checkpoint without ever exposing a partial pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != "V16_STAGE_CHECKPOINT":
        raise RuntimeError(f"Invalid V16 checkpoint: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    """Return a content hash for inputs that define a checkpoint's validity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(config_path: Path, reference_path: Path) -> dict[str, str]:
    """Identify scientific inputs required to reuse an analysis stage.

    The execution block is deliberately excluded: worker count and chunking
    alter wall-clock time only, while the deterministic input order, solver,
    mesh, tolerances, and all physical inputs remain unchanged.
    """
    scientific_config = load_config(config_path)
    scientific_config.pop("execution", None)
    scientific_bytes = json.dumps(
        scientific_config, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "study_id": str(scientific_config["study_id"]),
        "scientific_config_sha256": hashlib.sha256(scientific_bytes).hexdigest(),
        "reference_sha256": _sha256_file(reference_path),
    }


def _stage_checkpoint_path(checkpoint_dir: Path, stage: int) -> Path:
    return checkpoint_dir / f"stage_{int(stage):02d}.pkl"


def _persist_stage_checkpoint(
    checkpoint_dir: Path,
    stage: int,
    state: dict[str, Any],
    identity: dict[str, str],
    *,
    mode: str,
) -> None:
    """Atomically persist one completed analysis stage and its input identity."""
    if mode not in {"snapshot", "delta", "final"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")
    _atomic_pickle_dump({
        "schema": "V16_STAGE_CHECKPOINT",
        "completed_stage": int(stage),
        "checkpoint_mode": mode,
        "identity": identity,
        "state": state,
    }, _stage_checkpoint_path(checkpoint_dir, stage))
    print(f"[V16 checkpoint] stage {int(stage):02d} persisted", flush=True)


def _restore_checkpoint_state(
    checkpoint_dir: Path,
    identity: dict[str, str],
) -> tuple[dict[str, Any], int]:
    """Restore the highest contiguous, input-matched stage chain.

    Stages are stored as small deltas after an initial snapshot.  This avoids
    repeating completed calculations while keeping each checkpoint independently
    auditable.  The pre-existing stage-06 full snapshot is migrated in place so
    an interrupted V16 run never has to recompute its expensive first six stages.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    completed = 0
    first = _stage_checkpoint_path(checkpoint_dir, 1)
    legacy_six = _stage_checkpoint_path(checkpoint_dir, 6)
    if not first.exists() and legacy_six.exists():
        legacy = _load_checkpoint(legacy_six)
        if legacy is not None and int(legacy.get("completed_stage", -1)) == 6:
            legacy_state = legacy.get("state")
            if not isinstance(legacy_state, dict):
                raise RuntimeError("Legacy stage-06 checkpoint has no valid state payload")
            if legacy.get("identity") is None:
                state.update(legacy_state)
                completed = 6
                _persist_stage_checkpoint(checkpoint_dir, 6, dict(state), identity, mode="snapshot")
                print("[V16 resume] migrated validated legacy stage-06 checkpoint", flush=True)
            elif legacy.get("identity") == identity and legacy.get("checkpoint_mode") == "snapshot":
                state.update(legacy_state)
                completed = 6
            else:
                print("[V16 checkpoint] legacy stage-06 input identity mismatch; recomputing from stage 01", flush=True)

    for stage in range(completed + 1, 20):
        path = _stage_checkpoint_path(checkpoint_dir, stage)
        if not path.exists():
            break
        payload = _load_checkpoint(path)
        if payload is None:
            break
        if int(payload.get("completed_stage", -1)) != stage:
            raise RuntimeError(f"Checkpoint stage mismatch: {path}")
        if payload.get("identity") != identity:
            print(f"[V16 checkpoint] stage {stage:02d} input identity mismatch; recomputing from stage {completed + 1:02d}", flush=True)
            break
        delta = payload.get("state")
        if not isinstance(delta, dict):
            raise RuntimeError(f"Checkpoint state is invalid: {path}")
        mode = payload.get("checkpoint_mode")
        if mode == "snapshot":
            if completed not in {0, 6}:
                raise RuntimeError(f"Unexpected snapshot checkpoint after stage {completed}: {path}")
            state = dict(delta)
        elif mode == "delta":
            if completed != stage - 1:
                raise RuntimeError(f"Non-contiguous checkpoint chain at {path}")
            state.update(delta)
        else:
            raise RuntimeError(f"Unsupported restart checkpoint mode in {path}")
        completed = stage
    if completed:
        print(f"[V16 resume] restored validated checkpoint through stage {completed:02d}", flush=True)
    return state, completed


def run_all(config_path: str | Path, reference_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    reference_path = Path(reference_path)
    config = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out.parent / ".v16_restart_checkpoints"
    identity = _checkpoint_identity(config_path, reference_path)
    state, completed = _restore_checkpoint_state(checkpoint_dir, identity)

    def persist(stage: int, delta: dict[str, Any], *, mode: str = "delta") -> None:
        state.update(delta)
        _persist_stage_checkpoint(checkpoint_dir, stage, dict(delta) if mode == "delta" else dict(state), identity, mode=mode)

    if completed < 1:
        print("[V16 1/20] geometry and registered topology pool", flush=True)
        persist(1, {
            "pool": generate_geometry_pool(
                config, int(config["sampling"]["baseline_n"]), int(config["sampling"]["baseline_seed"])
            )
        }, mode="snapshot")
        completed = 1
    if completed < 2:
        print("[V16 2/20] conservative low-fidelity screen", flush=True)
        persist(2, {"screen": evaluate_geometry_pool(
            config_path, config, state["pool"], int(config["sampling"]["screen_n_cells"])
        )})
        completed = 2
    if completed < 3:
        print(f"[V16 3/20] all-design {int(config['sampling']['confirmation_n_cells'])}-cell confirmation", flush=True)
        unit = evaluate_geometry_pool(config_path, config, state["pool"], int(config["sampling"]["confirmation_n_cells"]))
        persist(3, {
            "unit": unit,
            "confirmation": screen_confirmation(state["screen"], unit),
            "ufront": unit_pareto(unit, config),
        })
        completed = 3
    if completed < 4:
        print("[V16 4/20] self-consistent fixed-service closure", flush=True)
        service = size_fixed_service(state["unit"], config)
        persist(4, {
            "service": service,
            "exact_sfront": exact_service_pareto(service, config),
            "sfront": decision_resolution_service_pareto(service, config),
        })
        completed = 4
    if completed < 5:
        print(f"[V16 5/20] {int(config['numerical_decision_validation']['refinement_n_cells'])}-cell retained-decision reclosure", flush=True)
        persist(5, {
            "grid_refinement": decision_grid_refinement(state["unit"], state["sfront"], config),
            "precision": pareto_precision_sensitivity(state["service"], config),
        })
        completed = 5
    if completed < 6:
        print("[V16 6/20] outlet-grade threshold sensitivity", flush=True)
        persist(6, {"threshold_sensitivity": outlet_grade_threshold_sensitivity(
            state["unit"], state["service"], config
        )})
        completed = 6
    if completed < 7:
        print("[V16 7/20] retained-design module-scale sensitivity", flush=True)
        persist(7, {"module_scale": module_scale_sensitivity(state["unit"], state["sfront"], config)})
        completed = 7
    if completed < 8:
        print("[V16 8/20] frozen-transport endpoint sensitivity", flush=True)
        persist(8, {"transport_sensitivity": transport_temperature_sensitivity(
            state["unit"], state["sfront"], config
        )})
        completed = 8
    if completed < 9:
        print("[V16 9/20] stratified capacity counterexample search", flush=True)
        monotonicity_probe = service_monotonicity_probe(state["unit"], state["service"], config)
        if not bool(monotonicity_probe["nondecreasing_from_previous_count"].all()):
            raise RuntimeError("V16 stratified capacity probe found non-monotone behavior")
        persist(9, {"monotonicity_probe": monotonicity_probe})
        completed = 9
    if completed < 10:
        print("[V16 10/20] full-integer retained-decision audit", flush=True)
        persist(10, {"monotonicity": service_monotonicity_audit(state["unit"], state["sfront"], config)})
        completed = 10
    if completed < 11:
        print("[V16 11/20] cross-scale advantage transfer", flush=True)
        persist(11, {
            "transfer": advantage_transfer(state["unit"], state["service"]),
            "pairwise": pairwise_scale_order(state["unit"], state["service"]),
            "interactions": interaction_indices(state["unit"]),
        })
        completed = 11
    if completed < 12:
        print("[V16 12/20] transferable external validation", flush=True)
        dlr_trace, dlr_metrics, dlr_cal = run_dlr_external_validation(
            config_path.parent.parent / config["verification"]["external_validation_file"]
        )
        persist(12, {"dlr_trace": dlr_trace, "dlr_metrics": dlr_metrics, "dlr_cal": dlr_cal})
        completed = 12
    if completed < 13:
        print("[V16 13/20] numerical and particle hierarchy verification", flush=True)
        persist(13, {"verification": run_grid_and_energy_verification(config), "particles": run_particle_hierarchy()})
        completed = 13
    if completed < 14:
        print("[V16 14/20] gradient-ratio and layer-fraction response", flush=True)
        persist(14, {"topology_response": topology_response_study(config_path, config)})
        completed = 14
    if completed < 15:
        print("[V16 15/20] independent sampling robustness", flush=True)
        persist(15, {"sampling": sampling_robustness(config_path, config)})
        completed = 15
    if completed < 16:
        print("[V16 16/20] bounded property stress", flush=True)
        persist(16, {"robust": bounded_robustness(
            config_path, config, state["pool"], state["unit"], state["service"]
        )})
        completed = 16
    if completed < 17:
        print("[V16 17/20] deterministic model-form stress", flush=True)
        persist(17, {"model_stress": model_form_stress(
            config_path, config, state["pool"], state["service"]
        )})
        completed = 17
    if completed < 18:
        print("[V16 18/20] standby-loss screen", flush=True)
        persist(18, {"standby": standby_loss_screen(config, state["unit"], state["service"])})
        completed = 18
    if completed < 19:
        print("[V16 19/20] reference control-volume closure", flush=True)
        persist(19, {"closure": reference_control_volume_closure(reference_path, config)})
        completed = 19

    pool, confirmation, unit, ufront = (state[k] for k in ("pool", "confirmation", "unit", "ufront"))
    service, exact_sfront, sfront = (state[k] for k in ("service", "exact_sfront", "sfront"))
    grid_refinement, precision, threshold_sensitivity = (state[k] for k in ("grid_refinement", "precision", "threshold_sensitivity"))
    module_scale, transport_sensitivity = (state[k] for k in ("module_scale", "transport_sensitivity"))
    monotonicity_probe, monotonicity = (state[k] for k in ("monotonicity_probe", "monotonicity"))
    transfer, pairwise, interactions = (state[k] for k in ("transfer", "pairwise", "interactions"))
    dlr_trace, dlr_metrics, dlr_cal = (state[k] for k in ("dlr_trace", "dlr_metrics", "dlr_cal"))
    verification, particles, topology_response = (state[k] for k in ("verification", "particles", "topology_response"))
    sampling, robust, model_stress, standby, closure = (
        state[k] for k in ("sampling", "robust", "model_stress", "standby", "closure")
    )

    outputs = {
        "geometry_pool.csv": pool,
        "screen_to_confirmation_v16.csv": confirmation,
        "codesign_results_v16.csv": unit,
        "unit_pareto_front_v16.csv": ufront,
        "fixed_service_results_v16.csv": service,
        "system_pareto_exact_audit_v16.csv": exact_sfront,
        "system_pareto_front_v16.csv": sfront,
        "decision_grid_refinement_v16.csv": grid_refinement,
        "pareto_precision_sensitivity_v16.csv": precision,
        "outlet_grade_threshold_sensitivity_v16.csv": threshold_sensitivity,
        "module_scale_sensitivity_v16.csv": module_scale,
        "transport_temperature_sensitivity_v16.csv": transport_sensitivity,
        "service_monotonicity_audit_v16.csv": monotonicity,
        "service_monotonicity_probe_v16.csv": monotonicity_probe,
        "advantage_transfer_v16.csv": transfer,
        "pairwise_scale_order_v16.csv": pairwise,
        "interaction_indices_v16.csv": interactions,
        "topology_response_v16.csv": topology_response,
        "external_validation_trace_v16.csv": dlr_trace,
        "external_validation_metrics_v16.csv": dlr_metrics,
        "external_validation_calibration_v16.csv": dlr_cal,
        "numerical_verification_v16.csv": verification,
        "particle_model_hierarchy_v16.csv": particles,
        "sampling_robustness_v16.csv": sampling,
        "bounded_robustness_v16.csv": robust,
        "model_form_stress_v16.csv": model_stress,
        "standby_loss_v16.csv": standby,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    (out / "reference_control_volume_closure_v16.json").write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
    t90_rel = confirmation["charge_t90_h_relative_difference_pct"].abs()
    hg_abs = confirmation["high_grade_discharge_fraction_absolute_difference"].abs()
    tie_groups = int(
        pairwise.groupby(["scenario_id", "base_design_id", "topology_id"])["system_exact_tie"]
        .any().sum()
    )
    power_limited = service[
        service["power_limited_unit_count"] > service["resolved_energy_limited_unit_count"]
    ]
    energy_limited = service.drop(power_limited.index)
    retention = 100.0 * transfer["ARI_material_to_system"]
    threshold_summary = []
    for (scenario_id, threshold), part in threshold_sensitivity.groupby(
        ["scenario_id", "threshold_sensitivity_grade_fraction"], sort=True
    ):
        threshold_summary.append({
            "scenario_id": str(scenario_id),
            "outlet_grade_threshold": float(threshold),
            "model_scope_admissible_cases": int(part["service_admissible"].sum()),
            "median_parallel_unit_count": float(part["parallel_unit_count"].median()),
            "median_parallel_unit_count_ratio_vs_g090": float(part["parallel_unit_count_ratio_vs_g090"].median()),
            "q05_parallel_unit_count_ratio_vs_g090": float(part["parallel_unit_count_ratio_vs_g090"].quantile(0.05)),
            "q95_parallel_unit_count_ratio_vs_g090": float(part["parallel_unit_count_ratio_vs_g090"].quantile(0.95)),
            "median_installed_volume_ratio_vs_g090": float(part["installed_volume_ratio_vs_g090"].median()),
        })
    module_scale_summary = []
    for factor, part in module_scale.groupby("linear_scale_factor", sort=True):
        module_scale_summary.append({
            "linear_scale_factor": float(factor),
            "retained_decision_cases": int(len(part)),
            "median_parallel_unit_count": float(part["parallel_unit_count"].median()),
            "minimum_parallel_unit_count": int(part["parallel_unit_count"].min()),
            "maximum_parallel_unit_count": int(part["parallel_unit_count"].max()),
            "median_unit_bed_pressure_drop_kPa": float(part["unit_bed_pressure_drop_kPa"].median()),
            "maximum_unit_bed_pressure_drop_kPa": float(part["unit_bed_pressure_drop_kPa"].max()),
            "median_bed_fraction_of_total_pressure_drop_pct": float(part["bed_fraction_of_total_pressure_drop_pct"].median()),
        })
    monotonic_case = monotonicity.groupby(["scenario_id", "reference_design_id"], sort=True).agg(
        all_increments_nondecreasing=("nondecreasing_from_previous_count", "all"),
        reported_minimum_matches_full_scan=("reported_minimum_matches_full_scan", "all"),
        scanned_counts=("parallel_unit_count", "size"),
    ).reset_index()
    grid_refinement_summary = {
        "retained_decision_cases": int(len(grid_refinement)),
        "production_n_cells": int(config["numerical_decision_validation"]["production_n_cells"]),
        "refinement_n_cells": int(config["numerical_decision_validation"]["refinement_n_cells"]),
        "maximum_absolute_deliverable_energy_change_pct": float(grid_refinement["deliverable_energy_change_pct"].abs().max()),
        "maximum_absolute_high_grade_fraction_change": float(grid_refinement["high_grade_fraction_change"].abs().max()),
        "maximum_absolute_parallel_unit_count_change_pct": float(grid_refinement["parallel_unit_count_change_pct"].abs().max()),
        "all_cases_accepted": bool(grid_refinement["grid_refinement_accepted"].all()),
        "scope": config["numerical_decision_validation"]["scope"],
    }
    probe_case_count = int(
        monotonicity_probe.groupby(["scenario_id", "reference_design_id"], sort=True).ngroups
    )
    cooling_holdout = dlr_metrics[dlr_metrics["role"].eq("validation")].iloc[0]
    threshold_hot_margin_K = (1.0 - float(config["service_scenarios"][0]["minimum_outlet_grade_fraction"])) * (
        float(config["common_duty_C"][1]) - float(config["common_duty_C"][0])
    )
    summary = {
        "study_id": config["study_id"],
        "version": config["version"],
        "n_base_geometries": int(config["sampling"]["baseline_n"]),
        "n_topology_designs": len(pool),
        "screen_n_cells": int(config["sampling"]["screen_n_cells"]),
        "confirmation_n_cells": int(config["sampling"]["confirmation_n_cells"]),
        "confirmation_scope": config["sampling"]["confirmation_scope"],
        "n_unit_cases": len(unit),
        "n_unit_admissible": int(unit["physics_comparison_admissible"].sum()),
        "n_unit_pareto": len(ufront),
        "n_fixed_service_cases": len(service),
        "n_fixed_service_admissible": int(service["service_admissible"].sum()),
        "n_system_pareto": len(sfront),
        "n_exact_system_pareto_audit": len(exact_sfront),
        "numerical_decision_validation": grid_refinement_summary,
        "service_closure": {
            "minimum_capacity_closure_error_pct": float(service["service_capacity_closure_error_pct"].min()),
            "maximum_capacity_closure_error_pct": float(service["service_capacity_closure_error_pct"].max()),
            "adjacent_lower_feasible_cases": int(service["adjacent_lower_count_feasible"].sum()),
            "median_resolved_to_nominal_unit_duty_ratio": float(service["resolved_to_nominal_unit_duty_ratio"].median()),
            "resolved_energy_count_increased_cases": int((service["resolved_energy_limited_unit_count"] > service["nominal_energy_limited_unit_count"]).sum()),
            "final_count_exceeds_nominal_energy_count_cases": int((service["parallel_unit_count"] > service["nominal_energy_limited_unit_count"]).sum()),
            "power_limited_cases": int(len(power_limited)),
            "power_limited_oversize_median_pct": float(power_limited["capacity_oversize_pct"].median()),
            "power_limited_oversize_max_pct": float(power_limited["capacity_oversize_pct"].max()),
            "energy_limited_cases": int(len(energy_limited)),
            "energy_limited_oversize_median_pct": float(energy_limited["capacity_oversize_pct"].median()),
            "energy_limited_oversize_max_pct": float(energy_limited["capacity_oversize_pct"].max()),
            "maximum_unit_bed_pressure_drop_kPa": float(service["unit_bed_pressure_drop_kPa"].max()),
            "median_header_to_bed_pressure_drop_ratio": float(
                (service["header_pressure_drop_kPa"] / service["unit_bed_pressure_drop_kPa"].clip(lower=1e-12)).median()
            ),
            "maximum_bed_fraction_of_unit_plus_header_pressure_drop_pct": float(
                (100.0 * service["unit_bed_pressure_drop_kPa"]
                 / (service["unit_bed_pressure_drop_kPa"] + service["header_pressure_drop_kPa"])).max()
            ),
        },
        "outlet_grade_threshold_sensitivity": threshold_summary,
        "module_scale_sensitivity": {
            "summary": module_scale_summary,
            "interpretation": config["module_scale_sensitivity"]["interpretation"],
        },
        "transport_temperature_sensitivity": {
            "retained_decision_cases": int(transport_sensitivity.groupby(["scenario_id", "reference_design_id"]).ngroups),
            "maximum_absolute_parallel_unit_count_change_pct": float(transport_sensitivity["parallel_unit_count_change_pct"].abs().max()),
            "maximum_absolute_installed_volume_change_pct": float(transport_sensitivity["installed_volume_change_pct"].abs().max()),
            "maximum_absolute_bed_pressure_drop_change_pct": float(transport_sensitivity["bed_pressure_drop_change_pct"].abs().max()),
        },
        "service_monotonicity_audit": {
            "retained_decision_cases": int(len(monotonic_case)),
            "integer_counts_scanned": int(len(monotonicity)),
            "cases_with_nondecreasing_installed_energy": int(monotonic_case["all_increments_nondecreasing"].sum()),
            "cases_matching_reported_minimum": int(monotonic_case["reported_minimum_matches_full_scan"].sum()),
            "scope": config["service_monotonicity_audit"]["scope"],
            "stratified_probe_cases": probe_case_count,
            "stratified_probe_count_evaluations": int(len(monotonicity_probe)),
            "stratified_probe_nondecreasing_cases": int(
                monotonicity_probe.groupby(["scenario_id", "reference_design_id"])["nondecreasing_from_previous_count"].all().sum()
            ),
        },
        "validation_threshold_context": {
            "registered_g090_hot_end_margin_K": float(threshold_hot_margin_K),
            "cooling_holdout_RMSE_K": float(cooling_holdout["RMSE_K"]),
            "cooling_holdout_mean_bias_K": float(cooling_holdout["mean_bias_K"]),
            "RMSE_to_threshold_margin_ratio": float(abs(cooling_holdout["RMSE_K"]) / threshold_hot_margin_K),
            "bias_to_threshold_margin_ratio": float(abs(cooling_holdout["mean_bias_K"]) / threshold_hot_margin_K),
            "interpretation": config["verification"]["threshold_interpretation"],
        },
        "pairwise_scale_order": {
            "matched_groups": int(pairwise.groupby(["scenario_id", "base_design_id", "topology_id"]).ngroups),
            "pairwise_comparisons": int(len(pairwise)),
            "pairwise_reversals": int((pairwise["pair_fate"] == "reversal").sum()),
            "system_exact_ties_from_nonzero_material_differences": int((pairwise["pair_fate"] == "system_exact_tie").sum()),
            "groups_with_at_least_one_tie": tie_groups,
            "groups_with_fully_strict_order": int(
                pairwise.groupby(["scenario_id", "base_design_id", "topology_id"]).ngroups
                - tie_groups
            ),
            "retention_min_pct": float(retention.min()),
            "retention_q05_pct": float(retention.quantile(0.05)),
            "retention_median_pct": float(retention.median()),
            "retention_q95_pct": float(retention.quantile(0.95)),
            "retention_max_pct": float(retention.max()),
        },
        "evidence_gate": {
            "engineering_deployment_eligible_unit_cases": int(unit["engineering_deployment_eligible"].sum()),
            "engineering_deployment_eligible_service_cases": int(service["engineering_deployment_eligible"].sum()),
            "interpretation": "Model-domain physics comparisons; not deployment-ready designs.",
        },
        "screen_confirmation": {
            "median_absolute_t90_relative_difference_pct": float(t90_rel.median()),
            "p95_absolute_t90_relative_difference_pct": float(t90_rel.quantile(0.95)),
            "median_high_grade_absolute_difference": float(hg_abs.median()),
            "p95_high_grade_absolute_difference": float(hg_abs.quantile(0.95)),
        },
        "external_validation": dlr_metrics.to_dict("records"),
        "claim_boundary": config["claim_boundary"],
        "status": "PASS",
    }
    (out / "analysis_summary_v16.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _persist_stage_checkpoint(
        checkpoint_dir,
        20,
        {"analysis_summary": summary, "result_table_names": sorted(outputs)},
        identity,
        mode="final",
    )
    print("[V16 20/20] result tables and analysis summary written", flush=True)
    return summary
