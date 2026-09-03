"""Verification and tiered independent validation for V16."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ltne_v16 import BedGeometryV16, evaluate_packed_bed_v16, simulate_air_benchmark_v16_fast
from .particle_radial_v16 import particle_model_hierarchy


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_dlr_external_validation(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate one heat-loss term on heating and validate the cooling segment.

    The input is a figure-level reconstruction of the open DLR data article.
    The raw Mendeley repository remains the preferred source for future blind
    validation. V16 therefore labels this as transferable model-class validation,
    not salt-specific validation and not a substitute for the raw data.
    """
    data = pd.read_csv(data_path)
    time_s = data["time_min"].to_numpy(float) * 60.0
    inlet = data["area_weighted_inlet_K"].to_numpy(float)
    observed = data["area_weighted_outlet_K"].to_numpy(float)
    mass_flow = np.where(data["time_min"].to_numpy(float) <= 210.0, 38.0 * 1.293 / 3600.0, 15.0 * 1.293 / 3600.0)
    heating = time_s <= 210.0 * 60.0
    cooling = ~heating
    calibration_rows: list[dict[str, float]] = []
    for wall_loss in np.arange(350.0, 501.0, 10.0):
        sim = simulate_air_benchmark_v16_fast(
            time_s=time_s,
            inlet_temperature_K=inlet,
            mass_flow_kg_s=mass_flow,
            n_cells=16,
            h_multiplier=1.0,
            axial_dispersion_m2_s=0.0,
            wall_loss_W_m3K=float(wall_loss),
            initial_temperature_K=300.0,
        )
        pred = np.asarray(sim["outlet_temperature_K"])
        calibration_rows.append({
            "wall_loss_W_m3K": wall_loss,
            "heating_RMSE_K": _rmse(pred[heating], observed[heating]),
            "heating_bias_K": float(np.mean(pred[heating] - observed[heating])),
            "n_cells": 16,
        })
    calibration = pd.DataFrame(calibration_rows).sort_values(["heating_RMSE_K", "wall_loss_W_m3K"])
    selected_wall = float(calibration.iloc[0]["wall_loss_W_m3K"])
    final = simulate_air_benchmark_v16_fast(
        time_s=time_s,
        inlet_temperature_K=inlet,
        mass_flow_kg_s=mass_flow,
        n_cells=40,
        h_multiplier=1.0,
        axial_dispersion_m2_s=0.0,
        wall_loss_W_m3K=selected_wall,
        initial_temperature_K=300.0,
    )
    pred = np.asarray(final["outlet_temperature_K"])
    lo = data["radial_min_K_outlet"].to_numpy(float)
    hi = data["radial_max_K_outlet"].to_numpy(float)
    digitisation_u = data["digitisation_temperature_uncertainty_K"].to_numpy(float)
    inside = (pred >= lo - digitisation_u) & (pred <= hi + digitisation_u)
    trace = pd.DataFrame({
        "time_min": data["time_min"],
        "phase": np.where(heating, "calibration_heating", "validation_cooling"),
        "inlet_temperature_K": inlet,
        "observed_area_weighted_outlet_K": observed,
        "observed_radial_min_K": lo,
        "observed_radial_max_K": hi,
        "predicted_1D_outlet_K": pred,
        "residual_area_weighted_K": pred - observed,
        "within_observed_radial_envelope_plus_digitisation_uncertainty": inside,
        "mass_flow_kg_s": mass_flow,
        "selected_wall_loss_W_m3K": selected_wall,
        "source_doi": "10.1016/j.dib.2025.111743",
        "dataset_doi": "10.17632/3pp86gdvh4.2",
    })
    metrics = pd.DataFrame([
        {
            "validation_id": "DLR_WEber_2025_HEATING_CALIBRATION",
            "role": "calibration",
            "observable": "area-weighted reconstructed outlet temperature",
            "RMSE_K": _rmse(pred[heating], observed[heating]),
            "NRMSE_pct_of_500K_span": 100.0 * _rmse(pred[heating], observed[heating]) / 500.0,
            "mean_bias_K": float(np.mean(pred[heating] - observed[heating])),
            "radial_envelope_coverage_pct": 100.0 * float(np.mean(inside[heating])),
            "selected_wall_loss_W_m3K": selected_wall,
            "n_cells": 40,
            "claim_boundary": "one heat-loss coefficient calibrated on heating; no heat-transfer multiplier or axial-dispersion calibration",
        },
        {
            "validation_id": "DLR_WEber_2025_COOLING_TEMPORAL_HOLDOUT",
            "role": "validation",
            "observable": "cooling outlet temperature and radial measurement envelope",
            "RMSE_K": _rmse(pred[cooling], observed[cooling]),
            "NRMSE_pct_of_500K_span": 100.0 * _rmse(pred[cooling], observed[cooling]) / 500.0,
            "mean_bias_K": float(np.mean(pred[cooling] - observed[cooling])),
            "radial_envelope_coverage_pct": 100.0 * float(np.mean(inside[cooling])),
            "selected_wall_loss_W_m3K": selected_wall,
            "n_cells": 40,
            "claim_boundary": "cooling held out; radial nonuniformity, wall inertia and other unresolved effects are reported as model-form discrepancy",
        },
        {
            "validation_id": "DLR_WEber_2025_FULL_TRACE",
            "role": "diagnostic",
            "observable": "full heating-cooling trace",
            "RMSE_K": _rmse(pred, observed),
            "NRMSE_pct_of_500K_span": 100.0 * _rmse(pred, observed) / 500.0,
            "mean_bias_K": float(np.mean(pred - observed)),
            "radial_envelope_coverage_pct": 100.0 * float(np.mean(inside)),
            "selected_wall_loss_W_m3K": selected_wall,
            "n_cells": 40,
            "claim_boundary": "figure-digitized benchmark with ±5 K digitization allowance; raw repository data are not redistributed",
        },
    ])
    return trace, metrics, calibration


def _advection_verification() -> pd.DataFrame:
    """Compare first-order upwind and MUSCL on a transported step."""
    rows: list[dict[str, float | str | int]] = []
    for n in (50, 100, 200):
        cfl = 0.4
        x = (np.arange(n) + 0.5) / n
        u0 = (x < 0.25).astype(float)
        n_steps = int(round(0.25 * n / cfl))
        up = u0.copy()
        tvd = u0.copy()
        for _ in range(n_steps):
            # periodic first-order upwind
            up = up - cfl * (up - np.roll(up, 1))
            left = tvd - np.roll(tvd, 1)
            right = np.roll(tvd, -1) - tvd
            slope = np.where(left * right > 0, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)
            flux = tvd + 0.5 * slope
            tvd = tvd - cfl * (flux - np.roll(flux, 1))
        exact = (((x - 0.25) % 1.0) < 0.25).astype(float)
        for name, pred in (("first_order_upwind", up), ("MUSCL_minmod", tvd)):
            rows.append({
                "verification_case": "periodic_step_advection",
                "method": name,
                "n_cells": n,
                "L1_error": float(np.mean(np.abs(pred - exact))),
                "transition_cells_0p1_to_0p9": int(np.sum((pred > 0.1) & (pred < 0.9))),
                "undershoot": float(min(0.0, pred.min())),
                "overshoot": float(max(0.0, pred.max() - 1.0)),
            })
    return pd.DataFrame(rows)


def run_grid_and_energy_verification(config: dict[str, Any]) -> pd.DataFrame:
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    g = config["reference_geometry"]
    rows: list[dict[str, Any]] = []
    for n in config["verification"]["grid_cells"]:
        for topology, ratio, orientation in (
            ("homogeneous", 1.0, "none"),
            ("graded_two_layer", 1.8, "fine_hot"),
        ):
            geom = BedGeometryV16(
                internal_diameter_m=float(g["internal_diameter_m"]),
                bed_height_m=float(g["bed_height_m"]),
                porosity=float(g["porosity"]),
                particle_diameter_m=float(g["particle_diameter_m"]),
                topology=topology,
                gradient_ratio=ratio,
                gradient_orientation=orientation,
                layer_fraction=0.5,
            )
            result = evaluate_packed_bed_v16(
                model_id="SOLAR_SALT_60_40_V10",
                design_id=f"GRID_{topology}_{n}",
                geometry=geom,
                mass_flow_kg_s=0.232,
                cold_temperature_K=cold_K,
                hot_temperature_K=hot_K,
                solid_density_kg_m3=float(g["solid_density_kg_m3"]),
                solid_cp_J_kgK=float(g["solid_cp_J_kgK"]),
                solid_thermal_conductivity_W_mK=float(g["solid_thermal_conductivity_W_mK"]),
                n_cells=int(n),
            )
            rows.append({
                "verification_case": "grid_energy_convergence",
                "method": "V16_MUSCL_LTNE",
                "topology": topology,
                "n_cells": int(n),
                "charge_t90_h": result["charge_t90_h"],
                "high_grade_discharge_fraction": result["high_grade_discharge_fraction"],
                "front_thickness_ratio": result["thermal_front_thickness_ratio"],
                "charge_energy_balance_error_pct": result["charge_energy_balance_error_pct"],
                "discharge_energy_balance_error_pct": result["discharge_energy_balance_error_pct"],
                "temperature_min_K": result["temperature_min_K"],
                "temperature_max_K": result["temperature_max_K"],
                "temperature_clipping_used": False,
            })
    frame = pd.DataFrame(rows)
    for topology, part in frame.groupby("topology"):
        finest = part.sort_values("n_cells").iloc[-1]
        idx = frame["topology"].eq(topology)
        frame.loc[idx, "t90_relative_error_vs_finest_pct"] = 100.0 * (
            frame.loc[idx, "charge_t90_h"] / finest["charge_t90_h"] - 1.0
        )
        frame.loc[idx, "high_grade_absolute_error_vs_finest"] = (
            frame.loc[idx, "high_grade_discharge_fraction"] - finest["high_grade_discharge_fraction"]
        )
    temporal_rows: list[dict[str, Any]] = []
    geom = BedGeometryV16(
        internal_diameter_m=float(g["internal_diameter_m"]),
        bed_height_m=float(g["bed_height_m"]),
        porosity=float(g["porosity"]),
        particle_diameter_m=float(g["particle_diameter_m"]),
    )
    production_cells = int(config["numerical_decision_validation"]["production_n_cells"])
    for maximum_dt in (5.0, 2.5, 1.25):
        result = evaluate_packed_bed_v16(
            model_id="SOLAR_SALT_60_40_V10",
            design_id=f"TIME_MAX_DT_{maximum_dt}",
            geometry=geom,
            mass_flow_kg_s=0.232,
            cold_temperature_K=cold_K,
            hot_temperature_K=hot_K,
            solid_density_kg_m3=float(g["solid_density_kg_m3"]),
            solid_cp_J_kgK=float(g["solid_cp_J_kgK"]),
            solid_thermal_conductivity_W_mK=float(g["solid_thermal_conductivity_W_mK"]),
            n_cells=production_cells,
            cfl_number=0.45,
            maximum_time_step_s=float(maximum_dt),
        )
        temporal_rows.append({
            "verification_case": "time_step_convergence",
            "method": "V16_operator_split_MUSCL_LTNE",
            "topology": "homogeneous",
            "n_cells": production_cells,
            "cfl_number": 0.45,
            "maximum_time_step_s": maximum_dt,
            "time_step_s": result["time_step_s"],
            "charge_t90_h": result["charge_t90_h"],
            "high_grade_discharge_fraction": result["high_grade_discharge_fraction"],
            "charge_energy_balance_error_pct": result["charge_energy_balance_error_pct"],
            "discharge_energy_balance_error_pct": result["discharge_energy_balance_error_pct"],
        })
    temporal = pd.DataFrame(temporal_rows).sort_values("maximum_time_step_s", ascending=False)
    reference = temporal.sort_values("maximum_time_step_s").iloc[0]
    temporal["t90_relative_error_vs_finest_pct"] = 100.0 * (
        temporal["charge_t90_h"] / reference["charge_t90_h"] - 1.0
    )
    temporal["high_grade_absolute_error_vs_finest"] = (
        temporal["high_grade_discharge_fraction"] - reference["high_grade_discharge_fraction"]
    )
    return pd.concat([frame, temporal, _advection_verification()], ignore_index=True, sort=False)


def run_particle_hierarchy() -> pd.DataFrame:
    rows = []
    for diameter in (0.008, 0.0126, 0.020):
        for h in (100.0, 300.0, 600.0):
            rows.append(particle_model_hierarchy(
                particle_diameter_m=diameter,
                density_kg_m3=2732.0,
                cp_J_kgK=1295.0,
                conductivity_W_mK=0.592,
                external_h_W_m2K=h,
                n_shells=16,
            ))
    return pd.DataFrame(rows)
