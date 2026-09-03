"""Conservative V16 packed-bed LTNE solver.

The production screen uses a monotonic MUSCL finite-volume advection
scheme with a minmod limiter, an exact local fluid-solid exchange step,
reverse flow on discharge, and explicit energy accounting.  No temperature
clipping is applied.  The solver supports homogeneous and two-layer graded
particle beds.  Axial dispersion and wall heat loss are opt-in model-hierarchy
terms rather than hidden calibration factors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .fluid_properties import (
    fluid_properties,
    get_model_spec,
    volumetric_sensible_energy_J_m3,
)
from .crossscale_dynamic import ergun_pressure_drop_Pa, gunn_nusselt_number

J_PER_MWH = 3.6e9

try:
    from numba import njit
    SOLVER_BACKEND = "numba"
except ImportError:  # pragma: no cover
    SOLVER_BACKEND = "python_fallback_numba_unavailable"
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@dataclass(frozen=True)
class BedGeometryV16:
    internal_diameter_m: float
    bed_height_m: float
    porosity: float
    particle_diameter_m: float
    topology: str = "homogeneous"
    gradient_ratio: float = 1.0
    gradient_orientation: str = "none"
    layer_fraction: float = 0.5

    def validate(self) -> None:
        if self.internal_diameter_m <= 0 or self.bed_height_m <= 0:
            raise ValueError("bed dimensions must be positive")
        if not (0.30 < self.porosity < 0.70):
            raise ValueError("porosity outside conservative packed-bed range")
        if self.particle_diameter_m <= 0:
            raise ValueError("particle diameter must be positive")
        if self.topology not in {"homogeneous", "graded_two_layer"}:
            raise ValueError(f"unsupported topology: {self.topology}")
        if self.topology == "graded_two_layer":
            if not (1.05 <= self.gradient_ratio <= 3.0):
                raise ValueError("gradient_ratio must be in [1.05, 3.0]")
            if self.gradient_orientation not in {"fine_hot", "coarse_hot"}:
                raise ValueError("graded orientation must be fine_hot or coarse_hot")
            if not (0.20 <= self.layer_fraction <= 0.80):
                raise ValueError("layer_fraction must be in [0.20, 0.80]")


@njit(cache=True)
def _minmod(a: float, b: float) -> float:
    if a * b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@njit(cache=True)
def _advance_one_step(
    tf: np.ndarray,
    ts: np.ndarray,
    inlet_temperature: float,
    beta_adv: np.ndarray,
    cap_f: np.ndarray,
    cap_s: np.ndarray,
    hA: np.ndarray,
    alpha_f: np.ndarray,
    alpha_s: np.ndarray,
    wall_loss_W_m3K: float,
    ambient_temperature: float,
    dz: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One conservative operator-split step for positive flow.

    Arrays may be reversed by the caller to represent physical reverse flow.
    The advection step is conservative in the reduced constant-capacitance
    energy variable, and the exchange step is exact and locally conservative.
    """
    n = tf.size
    slope = np.empty(n, dtype=np.float64)
    flux = np.empty(n + 1, dtype=np.float64)
    tf_adv = np.empty(n, dtype=np.float64)
    for i in range(n):
        left = inlet_temperature if i == 0 else tf[i - 1]
        right = tf[i] if i == n - 1 else tf[i + 1]
        slope[i] = _minmod(tf[i] - left, right - tf[i])
    flux[0] = inlet_temperature
    for j in range(1, n):
        flux[j] = tf[j - 1] + 0.5 * slope[j - 1]
    flux[n] = tf[n - 1]
    for i in range(n):
        tf_adv[i] = tf[i] - dt * beta_adv[i] * (flux[i + 1] - flux[i])

    # Optional axial dispersion/conduction, zero-gradient boundaries.
    if np.max(alpha_f) > 0.0 or np.max(alpha_s) > 0.0:
        tf_diff = tf_adv.copy()
        ts_diff = ts.copy()
        for i in range(n):
            lf = tf_adv[0] if i == 0 else tf_adv[i - 1]
            rf = tf_adv[n - 1] if i == n - 1 else tf_adv[i + 1]
            ls = ts[0] if i == 0 else ts[i - 1]
            rs = ts[n - 1] if i == n - 1 else ts[i + 1]
            tf_diff[i] = tf_adv[i] + dt * alpha_f[i] * (lf - 2.0 * tf_adv[i] + rf) / (dz * dz)
            ts_diff[i] = ts[i] + dt * alpha_s[i] * (ls - 2.0 * ts[i] + rs) / (dz * dz)
        tf_adv = tf_diff
        ts_work = ts_diff
    else:
        ts_work = ts

    # Exact two-capacitance interphase exchange.
    tf_new = np.empty(n, dtype=np.float64)
    ts_new = np.empty(n, dtype=np.float64)
    for i in range(n):
        cf = cap_f[i]
        cs = cap_s[i]
        ctot = cf + cs
        mean_t = (cf * tf_adv[i] + cs * ts_work[i]) / ctot
        decay = math.exp(-hA[i] * (1.0 / cf + 1.0 / cs) * dt)
        lag = (tf_adv[i] - ts_work[i]) * decay
        tf_new[i] = mean_t + (cs / ctot) * lag
        ts_new[i] = mean_t - (cf / ctot) * lag

    # Wall loss acts on the local combined inventory and preserves LTNE lag.
    loss_density_J_m3 = 0.0
    if wall_loss_W_m3K > 0.0:
        for i in range(n):
            ctot = cap_f[i] + cap_s[i]
            mean_before = (cap_f[i] * tf_new[i] + cap_s[i] * ts_new[i]) / ctot
            mean_after = ambient_temperature + (mean_before - ambient_temperature) * math.exp(
                -wall_loss_W_m3K * dt / ctot
            )
            shift = mean_after - mean_before
            tf_new[i] += shift
            ts_new[i] += shift
            loss_density_J_m3 += ctot * (mean_before - mean_after)
    return tf_new, ts_new, loss_density_J_m3


@njit(cache=True)
def _front_metrics_numba(ts: np.ndarray, cold: float, hot: float, dz: float) -> tuple:
    """Measure a single axial thermocline and flag an ambiguous profile.

    The retained thickness is only defined when each registered isotherm has
    one crossing.  The crossing counts and monotonicity violations accompany
    the result so a first-crossing value can never silently stand in for a
    multi-front or oscillatory profile.
    """
    n = ts.size
    z90 = -1.0
    z10 = -1.0
    n90 = 0
    n10 = 0
    nonmonotonic_steps = 0
    span = hot - cold
    for i in range(n - 1):
        th0 = (ts[i] - cold) / span
        th1 = (ts[i + 1] - cold) / span
        if th1 > th0 + 1.0e-10:
            nonmonotonic_steps += 1
        if (th0 >= 0.9 and th1 < 0.9) or (th0 <= 0.9 and th1 > 0.9):
            n90 += 1
            if z90 < 0.0:
                z90 = (i + 0.5) * dz + (0.9 - th0) / (th1 - th0) * dz
        if (th0 >= 0.1 and th1 < 0.1) or (th0 <= 0.1 and th1 > 0.1):
            n10 += 1
            if z10 < 0.0:
                z10 = (i + 0.5) * dz + (0.1 - th0) / (th1 - th0) * dz
    thickness = -1.0
    if n90 == 1 and n10 == 1:
        thickness = abs(z10 - z90)
    return thickness, n90, n10, nonmonotonic_steps


@njit(cache=True)
def _run_constant_boundary_core(
    mode_charge: int,
    reverse_profile: int,
    n_steps: int,
    dt: float,
    dz: float,
    area: float,
    cell_volume: float,
    cold: float,
    hot: float,
    inlet: float,
    initial_fluid_physical: np.ndarray,
    initial_solid_physical: np.ndarray,
    mass_flow: float,
    cp_flux: float,
    beta_adv_physical: np.ndarray,
    cap_f_physical: np.ndarray,
    cap_s_physical: np.ndarray,
    hA_physical: np.ndarray,
    alpha_f_physical: np.ndarray,
    alpha_s_physical: np.ndarray,
    wall_loss_W_m3K: float,
    ambient_temperature: float,
    q_span: float,
    minimum_outlet_grade_fraction: float,
) -> tuple:
    n = cap_f_physical.size
    if reverse_profile == 1:
        beta_adv = beta_adv_physical[::-1].copy()
        cap_f = cap_f_physical[::-1].copy()
        cap_s = cap_s_physical[::-1].copy()
        hA = hA_physical[::-1].copy()
        alpha_f = alpha_f_physical[::-1].copy()
        alpha_s = alpha_s_physical[::-1].copy()
        initial_fluid = initial_fluid_physical[::-1].copy()
        initial_solid = initial_solid_physical[::-1].copy()
    else:
        beta_adv = beta_adv_physical.copy()
        cap_f = cap_f_physical.copy()
        cap_s = cap_s_physical.copy()
        hA = hA_physical.copy()
        alpha_f = alpha_f_physical.copy()
        alpha_s = alpha_s_physical.copy()
        initial_fluid = initial_fluid_physical.copy()
        initial_solid = initial_solid_physical.copy()

    tf = np.empty(n, dtype=np.float64)
    ts = np.empty(n, dtype=np.float64)
    for i in range(n):
        tf[i] = initial_fluid[i]
        ts[i] = initial_solid[i]
    t10 = -1.0
    t50 = -1.0
    t90 = -1.0
    front50 = -1.0
    front_crossings_90 = -1
    front_crossings_10 = -1
    front_nonmonotonic_steps = -1
    useful = 0.0
    boundary_energy = 0.0
    wall_loss = 0.0
    high_grade_duration = 0.0
    # Extrema cover the initial condition and every subsequent transient
    # state.  They are deliberately not inferred from the terminal fields.
    transient_tf_min = np.min(tf)
    transient_tf_max = np.max(tf)
    transient_ts_min = np.min(ts)
    transient_ts_max = np.max(ts)
    high_grade = cold + minimum_outlet_grade_fraction * (hot - cold)
    time_s = 0.0
    initial_inventory = 0.0
    for i in range(n):
        initial_inventory += (cap_f[i] * (tf[i] - cold) + cap_s[i] * (ts[i] - cold)) * cell_volume
    inventory = initial_inventory

    for _ in range(n_steps):
        outlet_before = tf[n - 1]
        tf, ts, loss_density = _advance_one_step(
            tf, ts, inlet, beta_adv, cap_f, cap_s, hA,
            alpha_f, alpha_s, wall_loss_W_m3K, ambient_temperature, dz, dt,
        )
        time_s += dt
        wall_loss += loss_density * cell_volume
        outlet = tf[n - 1]
        # Old-time face-flux quadrature, signed positive into the bed for
        # charge and positive out of the bed for discharge.
        if mode_charge == 1:
            p0 = mass_flow * cp_flux * (inlet - outlet_before)
            p1 = mass_flow * cp_flux * (inlet - outlet)
        else:
            p0 = mass_flow * cp_flux * (outlet_before - inlet)
            p1 = mass_flow * cp_flux * (outlet - inlet)
        # The explicit finite-volume update uses the old-time face fluxes;
        # matching that quadrature yields a discrete energy identity.
        step_boundary = p0 * dt
        boundary_energy += step_boundary

        transient_tf_min = min(transient_tf_min, np.min(tf))
        transient_tf_max = max(transient_tf_max, np.max(tf))
        transient_ts_min = min(transient_ts_min, np.min(ts))
        transient_ts_max = max(transient_ts_max, np.max(ts))

        inventory = 0.0
        for i in range(n):
            inventory += (cap_f[i] * (tf[i] - cold) + cap_s[i] * (ts[i] - cold)) * cell_volume
        if mode_charge == 1:
            frac = inventory / q_span
            if t10 < 0.0 and frac >= 0.1:
                t10 = time_s
            if t50 < 0.0 and frac >= 0.5:
                t50 = time_s
                front50, front_crossings_90, front_crossings_10, front_nonmonotonic_steps = _front_metrics_numba(
                    ts, cold, hot, dz
                )
            if t90 < 0.0 and frac >= 0.9:
                t90 = time_s
        else:
            extracted = (initial_inventory - inventory) / max(initial_inventory, 1e-12)
            # Resolve the fraction of a time step that satisfies the outlet
            # threshold.  Counting the complete old-time step at a threshold
            # crossing biases deliverable energy by O(dt), which matters when
            # integer service closure is evaluated at fine resolution.  The
            # face energy itself retains the old-time quadrature required by
            # the discrete finite-volume energy identity.
            qualified_fraction = 0.0
            if outlet_before >= high_grade and outlet >= high_grade:
                qualified_fraction = 1.0
            elif (outlet_before >= high_grade) != (outlet >= high_grade):
                denominator = outlet - outlet_before
                if denominator != 0.0:
                    crossing_fraction = (high_grade - outlet_before) / denominator
                    crossing_fraction = min(max(crossing_fraction, 0.0), 1.0)
                    if outlet_before >= high_grade:
                        qualified_fraction = crossing_fraction
                    else:
                        qualified_fraction = 1.0 - crossing_fraction
            useful += max(step_boundary, 0.0) * qualified_fraction
            high_grade_duration += dt * qualified_fraction
            if t10 < 0.0 and extracted >= 0.1:
                t10 = time_s
            if t50 < 0.0 and extracted >= 0.5:
                t50 = time_s
            if t90 < 0.0 and extracted >= 0.9:
                t90 = time_s

    return (
        t10, t50, t90, front50, useful, boundary_energy, wall_loss,
        inventory, high_grade_duration, tf[n - 1], transient_tf_min, transient_tf_max,
        transient_ts_min, transient_ts_max, tf, ts,
        front_crossings_90, front_crossings_10, front_nonmonotonic_steps,
    )


def _layer_arrays(geometry: BedGeometryV16, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    geometry.validate()
    eps = np.full(n_cells, geometry.porosity, dtype=float)
    dp = np.full(n_cells, geometry.particle_diameter_m, dtype=float)
    if geometry.topology == "graded_two_layer":
        split = max(1, min(n_cells - 1, int(round(geometry.layer_fraction * n_cells))))
        ratio = geometry.gradient_ratio
        fine = geometry.particle_diameter_m / math.sqrt(ratio)
        coarse = geometry.particle_diameter_m * math.sqrt(ratio)
        if geometry.gradient_orientation == "fine_hot":
            dp[:split] = fine
            dp[split:] = coarse
        else:
            dp[:split] = coarse
            dp[split:] = fine
    return eps, dp


def _transport_arrays(
    *,
    model_id: str,
    mean_temperature_K: float,
    geometry: BedGeometryV16,
    mass_flow_kg_s: float,
    n_cells: int,
    solid_thermal_conductivity_W_mK: float,
    property_multipliers: dict[str, float] | None = None,
) -> dict[str, np.ndarray | float]:
    p = fluid_properties(model_id, mean_temperature_K)
    mult = {"density": 1.0, "cp": 1.0, "viscosity": 1.0, "conductivity": 1.0}
    if property_multipliers:
        unknown = set(property_multipliers) - set(mult)
        if unknown:
            raise ValueError(f"unknown property multipliers: {sorted(unknown)}")
        mult.update({k: float(v) for k, v in property_multipliers.items()})
    if not all(math.isfinite(v) and v > 0 for v in mult.values()):
        raise ValueError("property multipliers must be finite and positive")
    rho = p.density_kg_m3 * mult["density"]
    cp = p.cp_J_kgK * mult["cp"]
    mu = p.viscosity_Pa_s * mult["viscosity"]
    kf = p.thermal_conductivity_W_mK * mult["conductivity"]
    area = math.pi * geometry.internal_diameter_m**2 / 4.0
    eps, dp = _layer_arrays(geometry, n_cells)
    superficial = mass_flow_kg_s / (rho * area)
    hA = np.empty(n_cells)
    re = np.empty(n_cells)
    pr = np.full(n_cells, cp * mu / kf)
    nu = np.empty(n_cells)
    hconv = np.empty(n_cells)
    heff = np.empty(n_cells)
    bi = np.empty(n_cells)
    dp_grad = np.empty(n_cells)
    for i in range(n_cells):
        re[i] = rho * superficial * dp[i] / mu
        nu[i] = gunn_nusselt_number(float(re[i]), float(pr[i]), float(eps[i]))
        hconv[i] = nu[i] * kf / dp[i]
        heff[i] = 1.0 / (1.0 / hconv[i] + dp[i] / (10.0 * solid_thermal_conductivity_W_mK))
        bi[i] = hconv[i] * (dp[i] / 6.0) / solid_thermal_conductivity_W_mK
        hA[i] = heff[i] * 6.0 * (1.0 - eps[i]) / dp[i]
        visc = 150.0 * mu * superficial * (1.0 - eps[i]) ** 2 / (eps[i] ** 3 * dp[i] ** 2)
        iner = 1.75 * rho * superficial**2 * (1.0 - eps[i]) / (eps[i] ** 3 * dp[i])
        dp_grad[i] = visc + iner
    return {
        "rho": rho, "cp": cp, "mu": mu, "kf": kf, "area": area,
        "eps": eps, "dp": dp, "superficial": superficial, "hA": hA,
        "re": re, "pr": pr, "nu": nu, "hconv": hconv, "heff": heff,
        "bi": bi, "dp_grad": dp_grad, "multipliers": mult,
    }


def evaluate_packed_bed_v16(
    *,
    model_id: str,
    design_id: str,
    geometry: BedGeometryV16,
    mass_flow_kg_s: float,
    cold_temperature_K: float,
    hot_temperature_K: float,
    solid_density_kg_m3: float,
    solid_cp_J_kgK: float,
    solid_thermal_conductivity_W_mK: float,
    n_cells: int = 64,
    axial_dispersion_m2_s: float = 0.0,
    solid_axial_diffusivity_m2_s: float = 0.0,
    wall_loss_W_m3K: float = 0.0,
    ambient_temperature_K: float | None = None,
    minimum_outlet_grade_fraction: float = 0.9,
    cfl_number: float = 0.45,
    maximum_time_step_s: float = 5.0,
    property_multipliers: dict[str, float] | None = None,
    transport_property_temperature_K: float | None = None,
) -> dict[str, Any]:
    """Evaluate one unit with conservative charge and counter-flow discharge."""
    geometry.validate()
    if n_cells < 8:
        raise ValueError("V16 requires at least eight axial cells")
    if mass_flow_kg_s <= 0 or hot_temperature_K <= cold_temperature_K:
        raise ValueError("non-physical flow or temperature span")
    if not (0.0 < minimum_outlet_grade_fraction < 1.0):
        raise ValueError("minimum_outlet_grade_fraction must be between zero and one")
    if not (0.0 < cfl_number <= 0.5) or maximum_time_step_s <= 0.0:
        raise ValueError("invalid time-step control")
    # Fail closed at both temperature endpoints.
    fluid_properties(model_id, cold_temperature_K)
    fluid_properties(model_id, hot_temperature_K)
    mean_T = 0.5 * (cold_temperature_K + hot_temperature_K)
    transport_T = mean_T if transport_property_temperature_K is None else float(transport_property_temperature_K)
    if not (cold_temperature_K <= transport_T <= hot_temperature_K):
        raise ValueError("transport_property_temperature_K must lie within the registered duty")
    tr = _transport_arrays(
        model_id=model_id, mean_temperature_K=transport_T, geometry=geometry,
        mass_flow_kg_s=mass_flow_kg_s, n_cells=n_cells,
        solid_thermal_conductivity_W_mK=solid_thermal_conductivity_W_mK,
        property_multipliers=property_multipliers,
    )
    area = float(tr["area"])
    eps = np.asarray(tr["eps"], dtype=float)
    dp = np.asarray(tr["dp"], dtype=float)
    hA = np.asarray(tr["hA"], dtype=float)
    rho = float(tr["rho"])
    cp_flux = float(tr["cp"])
    height = geometry.bed_height_m
    dz = height / n_cells
    cell_volume = area * dz
    volume = area * height
    delta_t = hot_temperature_K - cold_temperature_K

    fluid_energy_density = volumetric_sensible_energy_J_m3(model_id, cold_temperature_K, hot_temperature_K)
    mult = tr["multipliers"]
    fluid_energy_density *= float(mult["density"]) * float(mult["cp"])
    rho_cp_eff = fluid_energy_density / delta_t
    cap_f = eps * rho_cp_eff
    cap_s = (1.0 - eps) * solid_density_kg_m3 * solid_cp_J_kgK
    beta_adv = mass_flow_kg_s * cp_flux / (area * dz * cap_f)
    alpha_f = np.full(n_cells, max(float(axial_dispersion_m2_s), 0.0), dtype=float)
    alpha_s = np.full(n_cells, max(float(solid_axial_diffusivity_m2_s), 0.0), dtype=float)
    q_span = float(np.sum((cap_f + cap_s) * cell_volume) * delta_t)
    nominal_s = q_span / max(mass_flow_kg_s * cp_flux * delta_t, 1e-12)

    dt_limits = [float(maximum_time_step_s), float(cfl_number) / float(np.max(beta_adv))]
    if np.max(alpha_f) > 0:
        dt_limits.append(0.42 * dz * dz / float(np.max(alpha_f)))
    if np.max(alpha_s) > 0:
        dt_limits.append(0.42 * dz * dz / float(np.max(alpha_s)))
    dt_target = max(1e-4, min(dt_limits))
    t_end = 2.2 * nominal_s
    n_steps = max(2, int(math.ceil(t_end / dt_target)))
    dt = t_end / n_steps
    if wall_loss_W_m3K > 0.0 and ambient_temperature_K is None:
        raise ValueError(
            "ambient_temperature_K must be explicitly registered whenever wall_loss_W_m3K is nonzero"
        )
    ambient = cold_temperature_K if ambient_temperature_K is None else float(ambient_temperature_K)

    cold_initial = np.full(n_cells, cold_temperature_K, dtype=np.float64)
    charge = _run_constant_boundary_core(
        1, 0, n_steps, dt, dz, area, cell_volume,
        cold_temperature_K, hot_temperature_K, hot_temperature_K, cold_initial, cold_initial,
        mass_flow_kg_s, cp_flux, beta_adv, cap_f, cap_s, hA, alpha_f, alpha_s,
        float(wall_loss_W_m3K), ambient, q_span, float(minimum_outlet_grade_fraction),
    )
    # Counter-flow discharge inherits the actual charge-end temperature field.
    # The physical profile and local coefficients are reversed together so that
    # the positive computational direction remains the discharge-flow direction.
    discharge = _run_constant_boundary_core(
        0, 1, n_steps, dt, dz, area, cell_volume,
        cold_temperature_K, hot_temperature_K, cold_temperature_K, charge[14], charge[15],
        mass_flow_kg_s, cp_flux, beta_adv, cap_f, cap_s, hA, alpha_f, alpha_s,
        float(wall_loss_W_m3K), ambient, q_span, float(minimum_outlet_grade_fraction),
    )

    charge_inventory = float(charge[7])
    discharge_inventory = float(discharge[7])
    charge_input = float(charge[5])
    discharge_output = float(discharge[5])
    charge_wall = float(charge[6])
    discharge_wall = float(discharge[6])
    charge_balance = 100.0 * ((charge_inventory + charge_wall) / max(charge_input, 1e-12) - 1.0)
    discharge_balance = 100.0 * (
        (discharge_output + discharge_inventory + discharge_wall) / max(charge_inventory, 1e-12) - 1.0
    )
    pressure_drop = float(np.sum(np.asarray(tr["dp_grad"]) * dz))
    pump_power = pressure_drop * (mass_flow_kg_s / rho)
    capacity = q_span / J_PER_MWH
    t90 = None if charge[2] < 0 else float(charge[2])
    spec = get_model_spec(model_id)
    salt_mass = float(np.sum(eps * cell_volume) * rho)
    solid_mass = float(np.sum((1.0 - eps) * cell_volume) * solid_density_kg_m3)
    gamma = float(np.sum(cap_f) / np.sum(cap_s))
    interphase_ntu = float(np.sum(hA * cell_volume) / max(mass_flow_kg_s * cp_flux, 1e-12))
    result = {
        "design_id": design_id,
        "study_model": "V16_conservative_MUSCL_LTNE_effective_particle_resistance",
        "solver_backend": SOLVER_BACKEND,
        "fluid_model_id": model_id,
        "fluid_name": spec.material_name,
        "family": spec.family,
        "topology": geometry.topology,
        "gradient_ratio": geometry.gradient_ratio,
        "gradient_orientation": geometry.gradient_orientation,
        "layer_fraction": geometry.layer_fraction,
        "counterflow_discharge": True,
        "cycle_coupled_discharge": True,
        "discharge_initialization": "charge_end_profile_reversed_for_counterflow",
        "temperature_clipping_used": False,
        "internal_diameter_m": geometry.internal_diameter_m,
        "bed_height_m": geometry.bed_height_m,
        "porosity": geometry.porosity,
        "particle_diameter_m": geometry.particle_diameter_m,
        "particle_diameter_min_m": float(np.min(dp)),
        "particle_diameter_max_m": float(np.max(dp)),
        "mass_flow_kg_s": mass_flow_kg_s,
        "nominal_thermal_input_kW": mass_flow_kg_s * cp_flux * delta_t / 1000.0,
        "cold_temperature_C": cold_temperature_K - 273.15,
        "hot_temperature_C": hot_temperature_K - 273.15,
        "n_cells": n_cells,
        "time_step_s": dt,
        "cfl_number": float(cfl_number),
        "maximum_time_step_s": float(maximum_time_step_s),
        "transport_property_temperature_C": transport_T - 273.15,
        "transport_property_treatment": "run-wise frozen at the registered transport-property temperature",
        "axial_dispersion_m2_s": float(axial_dispersion_m2_s),
        "solid_axial_diffusivity_m2_s": float(solid_axial_diffusivity_m2_s),
        "wall_loss_W_m3K": float(wall_loss_W_m3K),
        "ambient_temperature_C": None if ambient_temperature_K is None else ambient - 273.15,
        "temperature_extrema_scope": "initial state and every explicit transient time step",
        "storage_volume_m3": volume,
        "salt_inventory_kg": salt_mass,
        "solid_inventory_kg": solid_mass,
        "stored_heat_MWh": capacity,
        "volumetric_capacity_MWh_m3": capacity / volume,
        "fluid_volumetric_sensible_kWh_m3": fluid_energy_density / 3.6e6,
        "charge_t10_h": None if charge[0] < 0 else float(charge[0]) / 3600.0,
        "charge_t50_h": None if charge[1] < 0 else float(charge[1]) / 3600.0,
        "charge_t90_h": None if charge[2] < 0 else float(charge[2]) / 3600.0,
        "discharge_t10_h": None if discharge[0] < 0 else float(discharge[0]) / 3600.0,
        "discharge_t50_h": None if discharge[1] < 0 else float(discharge[1]) / 3600.0,
        "discharge_t90_h": None if discharge[2] < 0 else float(discharge[2]) / 3600.0,
        "high_grade_discharge_duration_h": float(discharge[8]) / 3600.0,
        "high_grade_discharge_fraction": float(discharge[4] / max(q_span, 1e-12)),
        "high_grade_fraction_of_charged_inventory": float(discharge[4] / max(charge_inventory, 1e-12)),
        "deliverable_energy_MWh": float(discharge[4]) / J_PER_MWH,
        "minimum_outlet_grade_fraction": float(minimum_outlet_grade_fraction),
        "deliverable_energy_definition": "time integral of positive outlet enthalpy flow following the simulated charge end state, with linear within-step interpolation at the registered outlet-grade threshold",
        "threshold_crossing_interpolation_used": True,
        "thermal_front_thickness_ratio": None if charge[3] < 0 else float(charge[3]) / height,
        "thermal_front_crossings_90": int(charge[16]),
        "thermal_front_crossings_10": int(charge[17]),
        "thermal_front_nonmonotonic_steps": int(charge[18]),
        "thermal_front_definition": "0.90-to-0.10 solid-temperature thickness at charge t50; reported only for a single crossing of each isotherm",
        "charge_energy_balance_error_pct": charge_balance,
        "discharge_energy_balance_error_pct": discharge_balance,
        "charge_final_inventory_fraction": charge_inventory / max(q_span, 1e-12),
        "discharge_energy_recovery_fraction": discharge_output / max(charge_inventory, 1e-12),
        "charge_wall_loss_fraction": charge_wall / max(charge_input, 1e-12),
        "discharge_wall_loss_fraction": discharge_wall / max(charge_inventory, 1e-12),
        "pressure_drop_kPa": pressure_drop / 1000.0,
        "pump_power_W": pump_power,
        "pump_energy_fraction": None if t90 is None else pump_power * t90 / max(q_span, 1e-12),
        "particle_Re_min": float(np.min(tr["re"])),
        "particle_Re_max": float(np.max(tr["re"])),
        "Pr": float(np.mean(tr["pr"])),
        "Nu_min": float(np.min(tr["nu"])),
        "Nu_max": float(np.max(tr["nu"])),
        "particle_Biot_raw_min": float(np.min(tr["bi"])),
        "particle_Biot_raw_max": float(np.max(tr["bi"])),
        "particle_resistance_correction_min": float(np.min(tr["heff"] / tr["hconv"])),
        "particle_resistance_correction_max": float(np.max(tr["heff"] / tr["hconv"])),
        "thermal_capacity_ratio_Gamma": gamma,
        "interphase_NTU": interphase_ntu,
        "bed_to_particle_ratio_min": geometry.bed_height_m / float(np.max(dp)),
        "temperature_min_K": min(float(charge[10]), float(charge[12]), float(discharge[10]), float(discharge[12])),
        "temperature_max_K": max(float(charge[11]), float(charge[13]), float(discharge[11]), float(discharge[13])),
        "correlation_valid": bool(
            np.all((eps >= 0.35) & (eps < 1.0)) and np.max(tr["re"]) <= 1.0e5 and np.min(tr["pr"]) > 0
        ),
        "source_ids": ";".join(spec.source_ids),
        "quantitative_use_class": spec.quantitative_use_class,
        "density_multiplier": float(mult["density"]),
        "cp_multiplier": float(mult["cp"]),
        "viscosity_multiplier": float(mult["viscosity"]),
        "conductivity_multiplier": float(mult["conductivity"]),
    }
    return result


def simulate_time_varying_boundary_v16(
    *,
    inlet_time_s: np.ndarray,
    inlet_temperature_K: np.ndarray,
    mass_flow_time_kg_s: np.ndarray,
    initial_temperature_K: float,
    geometry: BedGeometryV16,
    fluid_density_kg_m3: float,
    fluid_cp_J_kgK: float,
    fluid_viscosity_Pa_s: float,
    fluid_conductivity_W_mK: float,
    solid_density_kg_m3: float,
    solid_cp_function: Callable[[np.ndarray], np.ndarray],
    solid_conductivity_function: Callable[[np.ndarray], np.ndarray],
    n_cells: int = 80,
    h_multiplier: float = 1.0,
    axial_dispersion_m2_s: float = 0.0,
    wall_loss_W_m3K: float = 0.0,
    ambient_temperature_K: float = 298.15,
    record_interval_s: float = 60.0,
) -> dict[str, np.ndarray | float]:
    """Inspectable model-form validation branch for a prescribed experiment.

    The external air benchmark uses measured inlet-temperature and flow histories.
    Air properties may be supplied at the representative state; particle Cp and k
    are updated from the source equations at each step.  This branch is deliberately
    separate from the molten-salt design screen so calibration cannot leak into
    the design ranking.
    """
    geometry.validate()
    t_in = np.asarray(inlet_time_s, dtype=float)
    T_in = np.asarray(inlet_temperature_K, dtype=float)
    m_in = np.asarray(mass_flow_time_kg_s, dtype=float)
    if t_in.ndim != 1 or len(t_in) < 2 or not (len(t_in) == len(T_in) == len(m_in)):
        raise ValueError("boundary arrays must be one-dimensional and equal length")
    if not np.all(np.diff(t_in) > 0):
        raise ValueError("time grid must be strictly increasing")
    area = math.pi * geometry.internal_diameter_m**2 / 4.0
    eps, dp = _layer_arrays(geometry, n_cells)
    dz = geometry.bed_height_m / n_cells
    cell_volume = area * dz
    tf = np.full(n_cells, float(initial_temperature_K))
    ts = np.full(n_cells, float(initial_temperature_K))
    out_t: list[float] = [float(t_in[0])]
    out_T: list[float] = [float(tf[-1])]
    out_mean_s: list[float] = [float(np.mean(ts))]
    out_loss: list[float] = [0.0]
    next_record = float(t_in[0]) + record_interval_s
    cumulative_loss = 0.0
    current = float(t_in[0])
    end = float(t_in[-1])
    min_T = float(initial_temperature_K)
    max_T = float(initial_temperature_K)

    while current < end - 1e-9:
        inlet = float(np.interp(current, t_in, T_in))
        flow = max(float(np.interp(current, t_in, m_in)), 1e-9)
        solid_cp = np.asarray(solid_cp_function(ts), dtype=float)
        solid_k = np.asarray(solid_conductivity_function(ts), dtype=float)
        cap_f = eps * fluid_density_kg_m3 * fluid_cp_J_kgK
        cap_s = (1.0 - eps) * solid_density_kg_m3 * solid_cp
        superficial = flow / (fluid_density_kg_m3 * area)
        re = fluid_density_kg_m3 * superficial * dp / fluid_viscosity_Pa_s
        pr = fluid_cp_J_kgK * fluid_viscosity_Pa_s / fluid_conductivity_W_mK
        nu = np.array([gunn_nusselt_number(float(x), float(pr), float(e)) for x, e in zip(re, eps)])
        hconv = nu * fluid_conductivity_W_mK / dp
        heff = 1.0 / (1.0 / hconv + dp / (10.0 * solid_k))
        hA = h_multiplier * heff * 6.0 * (1.0 - eps) / dp
        beta = flow * fluid_cp_J_kgK / (area * dz * cap_f)
        alpha_f = np.full(n_cells, max(axial_dispersion_m2_s, 0.0))
        alpha_s = np.zeros(n_cells)
        limits = [2.0, 0.42 / float(np.max(beta)), end - current]
        if axial_dispersion_m2_s > 0:
            limits.append(0.40 * dz * dz / axial_dispersion_m2_s)
        dt = max(1e-4, min(limits))
        tf, ts, loss_density = _advance_one_step(
            tf, ts, inlet, beta, cap_f, cap_s, hA, alpha_f, alpha_s,
            wall_loss_W_m3K, ambient_temperature_K, dz, dt,
        )
        cumulative_loss += float(loss_density) * cell_volume
        current += dt
        min_T = min(min_T, float(tf.min()), float(ts.min()))
        max_T = max(max_T, float(tf.max()), float(ts.max()))
        if current + 1e-9 >= next_record or current >= end - 1e-9:
            out_t.append(current)
            out_T.append(float(tf[-1]))
            out_mean_s.append(float(np.average(ts, weights=(1.0 - eps))))
            out_loss.append(cumulative_loss)
            next_record += record_interval_s
    return {
        "time_s": np.asarray(out_t),
        "outlet_temperature_K": np.asarray(out_T),
        "mean_solid_temperature_K": np.asarray(out_mean_s),
        "cumulative_wall_loss_J": np.asarray(out_loss),
        "temperature_min_K": min_T,
        "temperature_max_K": max_T,
    }

@njit(cache=True)
def _air_particle_cp(T: float) -> float:
    return -8.1716e-9*T**4 + 2.1620e-5*T**3 - 2.1801e-2*T**2 + 10.3580*T - 8.8547e2


@njit(cache=True)
def _air_particle_k(T: float) -> float:
    # Source equation is W cm-1 K-1; convert to W m-1 K-1.
    return 100.0 * (3.6109e-12*T**4 - 9.7658e-9*T**3 + 1.0228e-5*T**2 - 5.1344e-3*T + 1.1864)


@njit(cache=True)
def _gunn_numba(re: float, pr: float, eps: float) -> float:
    a = 7.0 - 10.0*eps + 5.0*eps*eps
    b = 1.33 - 2.40*eps + 1.20*eps*eps
    return a*(1.0 + 0.7*re**0.2*pr**(1.0/3.0)) + b*re**0.7*pr**(1.0/3.0)


@njit(cache=True)
def _run_air_benchmark_core(
    time_s: np.ndarray,
    inlet_K: np.ndarray,
    mass_flow_kg_s: np.ndarray,
    n_cells: int,
    diameter_m: float,
    height_m: float,
    porosity: float,
    particle_diameter_m: float,
    initial_K: float,
    rho_air: float,
    cp_air: float,
    mu_air: float,
    k_air: float,
    rho_solid: float,
    h_multiplier: float,
    axial_dispersion: float,
    wall_loss: float,
    ambient_K: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    n_out = time_s.size
    area = math.pi*diameter_m*diameter_m/4.0
    dz = height_m/n_cells
    cell_volume = area*dz
    eps_arr = np.full(n_cells, porosity)
    tf = np.full(n_cells, initial_K)
    ts = np.full(n_cells, initial_K)
    out = np.empty(n_out)
    mean_s = np.empty(n_out)
    loss = np.empty(n_out)
    out[0] = initial_K
    mean_s[0] = initial_K
    loss[0] = 0.0
    cumulative_loss = 0.0
    min_temp = initial_K
    max_temp = initial_K
    current = time_s[0]
    output_index = 1
    schedule_index = 0
    alpha_f = np.full(n_cells, axial_dispersion)
    alpha_s = np.zeros(n_cells)
    while output_index < n_out:
        target_time = time_s[output_index]
        while current < target_time - 1e-9:
            while schedule_index < n_out-2 and current > time_s[schedule_index+1]:
                schedule_index += 1
            t0 = time_s[schedule_index]
            t1 = time_s[schedule_index+1]
            frac = 0.0 if t1 == t0 else (current-t0)/(t1-t0)
            inlet = inlet_K[schedule_index] + frac*(inlet_K[schedule_index+1]-inlet_K[schedule_index])
            flow = mass_flow_kg_s[schedule_index] + frac*(mass_flow_kg_s[schedule_index+1]-mass_flow_kg_s[schedule_index])
            if flow < 1e-10:
                flow = 1e-10
            mean_ts = 0.0
            for i in range(n_cells):
                mean_ts += ts[i]
            mean_ts /= n_cells
            cps = _air_particle_cp(mean_ts)
            ks = _air_particle_k(mean_ts)
            cap_f = np.full(n_cells, porosity*rho_air*cp_air)
            cap_s = np.full(n_cells, (1.0-porosity)*rho_solid*cps)
            superficial = flow/(rho_air*area)
            re = rho_air*superficial*particle_diameter_m/mu_air
            pr = cp_air*mu_air/k_air
            nu = _gunn_numba(re, pr, porosity)
            hconv = nu*k_air/particle_diameter_m
            heff = 1.0/(1.0/hconv + particle_diameter_m/(10.0*ks))
            hA = np.full(n_cells, h_multiplier*heff*6.0*(1.0-porosity)/particle_diameter_m)
            beta = np.full(n_cells, flow*cp_air/(area*dz*cap_f[0]))
            dt = 2.0
            cfl_dt = 0.42/beta[0]
            if cfl_dt < dt:
                dt = cfl_dt
            if axial_dispersion > 0.0:
                diff_dt = 0.40*dz*dz/axial_dispersion
                if diff_dt < dt:
                    dt = diff_dt
            if target_time-current < dt:
                dt = target_time-current
            tf, ts, loss_density = _advance_one_step(
                tf, ts, inlet, beta, cap_f, cap_s, hA, alpha_f, alpha_s,
                wall_loss, ambient_K, dz, dt,
            )
            cumulative_loss += loss_density*cell_volume
            current += dt
            for i in range(n_cells):
                if tf[i] < min_temp: min_temp = tf[i]
                if ts[i] < min_temp: min_temp = ts[i]
                if tf[i] > max_temp: max_temp = tf[i]
                if ts[i] > max_temp: max_temp = ts[i]
        out[output_index] = tf[n_cells-1]
        ssum = 0.0
        for i in range(n_cells): ssum += ts[i]
        mean_s[output_index] = ssum/n_cells
        loss[output_index] = cumulative_loss
        output_index += 1
    return out, mean_s, loss, min_temp, max_temp


def simulate_air_benchmark_v16(
    *,
    time_s: np.ndarray,
    inlet_temperature_K: np.ndarray,
    mass_flow_kg_s: np.ndarray,
    n_cells: int = 60,
    h_multiplier: float = 1.0,
    axial_dispersion_m2_s: float = 0.0,
    wall_loss_W_m3K: float = 0.0,
    initial_temperature_K: float = 300.0,
) -> dict[str, np.ndarray | float]:
    """Compiled DLR shallow-bed benchmark branch using published geometry/properties."""
    t = np.asarray(time_s, dtype=float)
    inlet = np.asarray(inlet_temperature_K, dtype=float)
    flow = np.asarray(mass_flow_kg_s, dtype=float)
    if not (t.ndim == inlet.ndim == flow.ndim == 1 and len(t) == len(inlet) == len(flow)):
        raise ValueError("benchmark arrays must be one-dimensional and equal length")
    out, mean_s, loss, tmin, tmax = _run_air_benchmark_core(
        t, inlet, flow, int(n_cells), 0.200, 0.100, 0.377, 0.00213,
        float(initial_temperature_K), 0.70, 1030.0, 2.8e-5, 0.044,
        3650.0, float(h_multiplier), float(axial_dispersion_m2_s),
        float(wall_loss_W_m3K), 298.15,
    )
    return {
        "time_s": t,
        "outlet_temperature_K": out,
        "mean_solid_temperature_K": mean_s,
        "cumulative_wall_loss_J": loss,
        "temperature_min_K": float(tmin),
        "temperature_max_K": float(tmax),
    }

@njit(cache=True)
def _run_air_benchmark_core2(
    time_s: np.ndarray,
    inlet_K: np.ndarray,
    mass_flow_kg_s: np.ndarray,
    n_cells: int,
    h_multiplier: float,
    axial_dispersion: float,
    wall_loss: float,
    initial_K: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    diameter_m=0.200; height_m=0.100; eps=0.377; dp=0.00213
    rho_air=0.70; cp_air=1030.0; mu_air=2.8e-5; k_air=0.044; rho_solid=3650.0; ambient=298.15
    n_out=time_s.size; area=math.pi*diameter_m*diameter_m/4.0; dz=height_m/n_cells; cell_volume=area*dz
    tf=np.full(n_cells,initial_K); ts=np.full(n_cells,initial_K)
    slope=np.empty(n_cells); flux=np.empty(n_cells+1); tf_adv=np.empty(n_cells); tf_diff=np.empty(n_cells); ts_diff=np.empty(n_cells); tf_new=np.empty(n_cells); ts_new=np.empty(n_cells)
    out=np.empty(n_out); mean_s=np.empty(n_out); loss=np.empty(n_out)
    out[0]=initial_K; mean_s[0]=initial_K; loss[0]=0.0
    current=time_s[0]; sched=0; cumloss=0.0; minT=initial_K; maxT=initial_K
    for oi in range(1,n_out):
        target=time_s[oi]
        while current < target-1e-9:
            while sched < n_out-2 and current > time_s[sched+1]: sched+=1
            frac=(current-time_s[sched])/(time_s[sched+1]-time_s[sched])
            inlet=inlet_K[sched]+frac*(inlet_K[sched+1]-inlet_K[sched])
            flow=mass_flow_kg_s[sched]+frac*(mass_flow_kg_s[sched+1]-mass_flow_kg_s[sched])
            if flow<1e-12: flow=1e-12
            mts=0.0
            for i in range(n_cells): mts+=ts[i]
            mts/=n_cells
            cps=_air_particle_cp(mts); ks=_air_particle_k(mts)
            cf=eps*rho_air*cp_air; cs=(1.0-eps)*rho_solid*cps
            superficial=flow/(rho_air*area); re=rho_air*superficial*dp/mu_air; pr=cp_air*mu_air/k_air
            nu=_gunn_numba(re,pr,eps); hconv=nu*k_air/dp; heff=1.0/(1.0/hconv+dp/(10.0*ks)); hA=h_multiplier*heff*6.0*(1.0-eps)/dp
            beta=flow*cp_air/(area*dz*cf)
            dt=2.0
            if 0.42/beta<dt: dt=0.42/beta
            if axial_dispersion>0.0 and 0.40*dz*dz/axial_dispersion<dt: dt=0.40*dz*dz/axial_dispersion
            if target-current<dt: dt=target-current
            # MUSCL advection
            for i in range(n_cells):
                left=inlet if i==0 else tf[i-1]; right=tf[i] if i==n_cells-1 else tf[i+1]
                slope[i]=_minmod(tf[i]-left,right-tf[i])
            flux[0]=inlet
            for j in range(1,n_cells): flux[j]=tf[j-1]+0.5*slope[j-1]
            flux[n_cells]=tf[n_cells-1]
            for i in range(n_cells): tf_adv[i]=tf[i]-dt*beta*(flux[i+1]-flux[i])
            # optional axial fluid dispersion
            if axial_dispersion>0.0:
                for i in range(n_cells):
                    left=tf_adv[0] if i==0 else tf_adv[i-1]; right=tf_adv[n_cells-1] if i==n_cells-1 else tf_adv[i+1]
                    tf_diff[i]=tf_adv[i]+dt*axial_dispersion*(left-2.0*tf_adv[i]+right)/(dz*dz)
            else:
                for i in range(n_cells): tf_diff[i]=tf_adv[i]
            # exact exchange and wall loss
            ctot=cf+cs; decay=math.exp(-hA*(1.0/cf+1.0/cs)*dt); loss_density=0.0
            for i in range(n_cells):
                mean=(cf*tf_diff[i]+cs*ts[i])/ctot; lag=(tf_diff[i]-ts[i])*decay
                a=mean+(cs/ctot)*lag; b=mean-(cf/ctot)*lag
                if wall_loss>0.0:
                    ma=(cf*a+cs*b)/ctot; mb=ambient+(ma-ambient)*math.exp(-wall_loss*dt/ctot); shift=mb-ma; a+=shift; b+=shift; loss_density+=ctot*(ma-mb)
                tf_new[i]=a; ts_new[i]=b
            for i in range(n_cells):
                tf[i]=tf_new[i]; ts[i]=ts_new[i]
                if tf[i]<minT: minT=tf[i]
                if ts[i]<minT: minT=ts[i]
                if tf[i]>maxT: maxT=tf[i]
                if ts[i]>maxT: maxT=ts[i]
            cumloss+=loss_density*cell_volume; current+=dt
        out[oi]=tf[n_cells-1]
        sm=0.0
        for i in range(n_cells): sm+=ts[i]
        mean_s[oi]=sm/n_cells; loss[oi]=cumloss
    return out,mean_s,loss,minT,maxT


def simulate_air_benchmark_v16_fast(
    *, time_s: np.ndarray, inlet_temperature_K: np.ndarray, mass_flow_kg_s: np.ndarray,
    n_cells: int=60, h_multiplier: float=1.0, axial_dispersion_m2_s: float=0.0,
    wall_loss_W_m3K: float=0.0, initial_temperature_K: float=300.0,
) -> dict[str,np.ndarray|float]:
    t=np.asarray(time_s,dtype=float); inlet=np.asarray(inlet_temperature_K,dtype=float); flow=np.asarray(mass_flow_kg_s,dtype=float)
    out,ms,loss,tmin,tmax=_run_air_benchmark_core2(t,inlet,flow,int(n_cells),float(h_multiplier),float(axial_dispersion_m2_s),float(wall_loss_W_m3K),float(initial_temperature_K))
    return {'time_s':t,'outlet_temperature_K':out,'mean_solid_temperature_K':ms,'cumulative_wall_loss_J':loss,'temperature_min_K':float(tmin),'temperature_max_K':float(tmax)}
