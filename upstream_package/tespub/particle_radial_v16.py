"""Radial finite-volume verification model for spherical storage particles."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


def spherical_particle_step_response(
    *,
    radius_m: float,
    density_kg_m3: float,
    cp_J_kgK: float,
    conductivity_W_mK: float,
    external_h_W_m2K: float,
    initial_temperature_K: float,
    boundary_temperature_K: float,
    n_shells: int = 12,
    t_end_s: float | None = None,
    n_output: int = 301,
) -> dict[str, np.ndarray | float]:
    if min(radius_m, density_kg_m3, cp_J_kgK, conductivity_W_mK, external_h_W_m2K) <= 0:
        raise ValueError("particle inputs must be positive")
    if n_shells < 3:
        raise ValueError("at least three radial shells are required")
    edges = np.linspace(0.0, radius_m, n_shells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    volumes = 4.0 / 3.0 * math.pi * (edges[1:]**3 - edges[:-1]**3)
    areas = 4.0 * math.pi * edges**2
    capacitance = density_kg_m3 * cp_J_kgK * volumes
    surface_area = 4.0 * math.pi * radius_m**2
    lumped_tau = density_kg_m3 * cp_J_kgK * (4.0 / 3.0 * math.pi * radius_m**3) / (
        external_h_W_m2K * surface_area
    )
    if t_end_s is None:
        t_end_s = 8.0 * lumped_tau * (1.0 + external_h_W_m2K * radius_m / conductivity_W_mK)
    t_eval = np.linspace(0.0, float(t_end_s), int(n_output))

    conductance = np.empty(n_shells - 1)
    for i in range(n_shells - 1):
        dr = centers[i + 1] - centers[i]
        conductance[i] = conductivity_W_mK * areas[i + 1] / dr

    def rhs(_t: float, T: np.ndarray) -> np.ndarray:
        q = np.zeros(n_shells)
        for i in range(n_shells - 1):
            transfer = conductance[i] * (T[i + 1] - T[i])
            q[i] += transfer
            q[i + 1] -= transfer
        q[-1] += external_h_W_m2K * surface_area * (boundary_temperature_K - T[-1])
        return q / capacitance

    sol = solve_ivp(rhs, (0.0, float(t_end_s)), np.full(n_shells, initial_temperature_K),
                    t_eval=t_eval, method="BDF", rtol=1e-8, atol=1e-9)
    if not sol.success:
        raise RuntimeError(sol.message)
    mean_T = np.average(sol.y, axis=0, weights=volumes)
    theta = (mean_T - initial_temperature_K) / (boundary_temperature_K - initial_temperature_K)
    surface_theta = (sol.y[-1] - initial_temperature_K) / (boundary_temperature_K - initial_temperature_K)
    center_theta = (sol.y[0] - initial_temperature_K) / (boundary_temperature_K - initial_temperature_K)
    t90 = float(np.interp(0.9, theta, sol.t)) if theta[-1] >= 0.9 else float("nan")
    return {
        "time_s": sol.t,
        "mean_theta": theta,
        "surface_theta": surface_theta,
        "center_theta": center_theta,
        "t90_s": t90,
        "Bi_radius": external_h_W_m2K * radius_m / conductivity_W_mK,
        "n_shells": n_shells,
    }


def particle_model_hierarchy(
    *,
    particle_diameter_m: float,
    density_kg_m3: float,
    cp_J_kgK: float,
    conductivity_W_mK: float,
    external_h_W_m2K: float,
    n_shells: int = 16,
) -> dict[str, float]:
    radius = 0.5 * particle_diameter_m
    volume = 4.0 / 3.0 * math.pi * radius**3
    area = 4.0 * math.pi * radius**2
    thermal_capacity = density_kg_m3 * cp_J_kgK * volume
    tau_lumped = thermal_capacity / (external_h_W_m2K * area)
    h_eff = 1.0 / (1.0 / external_h_W_m2K + particle_diameter_m / (10.0 * conductivity_W_mK))
    tau_resistance = thermal_capacity / (h_eff * area)
    radial = spherical_particle_step_response(
        radius_m=radius,
        density_kg_m3=density_kg_m3,
        cp_J_kgK=cp_J_kgK,
        conductivity_W_mK=conductivity_W_mK,
        external_h_W_m2K=external_h_W_m2K,
        initial_temperature_K=0.0,
        boundary_temperature_K=1.0,
        n_shells=n_shells,
    )
    t90_lumped = -math.log(0.1) * tau_lumped
    t90_resistance = -math.log(0.1) * tau_resistance
    return {
        "particle_diameter_m": particle_diameter_m,
        "external_h_W_m2K": external_h_W_m2K,
        "particle_conductivity_W_mK": conductivity_W_mK,
        "Bi_radius": external_h_W_m2K * radius / conductivity_W_mK,
        "t90_lumped_s": t90_lumped,
        "t90_effective_resistance_s": t90_resistance,
        "t90_radial_fv_s": float(radial["t90_s"]),
        "lumped_error_vs_radial_pct": 100.0 * (t90_lumped / float(radial["t90_s"]) - 1.0),
        "effective_resistance_error_vs_radial_pct": 100.0 * (t90_resistance / float(radial["t90_s"]) - 1.0),
        "n_radial_shells": n_shells,
    }
