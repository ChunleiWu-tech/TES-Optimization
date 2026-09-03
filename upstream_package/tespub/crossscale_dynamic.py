from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

J_PER_MWH = 3.6e9


@dataclass(frozen=True)
class SolarSaltProperties:
    temperature_C: float
    density_kg_m3: float
    cp_J_kgK: float
    viscosity_Pa_s: float
    thermal_conductivity_W_mK: float
    source_density_transport: str
    source_cp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def solar_salt_properties(temperature_K: float, cp_J_kgK: float = 1480.0) -> SolarSaltProperties:
    """Solar-Salt property set used by the reduced-order dynamic branch.

    Density, viscosity and conductivity are the published Solar-Salt correlations
    (60 wt% NaNO3 + 40 wt% KNO3) registered to Zavoico/Sandia SAND2001-2100.
    The heat capacity is deliberately passed from an experimental material state
    rather than invented for a nanoparticle mixture.
    """
    T_C = float(temperature_K) - 273.15
    if not (300.0 <= T_C <= 600.0):
        raise ValueError("Solar-Salt transport correlations are restricted here to the registered 300-600 C domain")
    rho = 2090.0 - 0.636 * T_C
    mu = 0.022714 - 0.120e-3 * T_C + 2.281e-7 * T_C**2 - 1.474e-10 * T_C**3
    k = 0.443 + 1.9e-4 * T_C
    if rho <= 0 or mu <= 0 or k <= 0 or cp_J_kgK <= 0:
        raise ValueError("Non-physical Solar-Salt property evaluation")
    return SolarSaltProperties(
        temperature_C=T_C,
        density_kg_m3=rho,
        cp_J_kgK=float(cp_J_kgK),
        viscosity_Pa_s=mu,
        thermal_conductivity_W_mK=k,
        source_density_transport="ZAVOICO_SANDIA_SOLAR_SALT_2001",
        source_cp="caller_supplied_experimental_state",
    )


def ergun_pressure_drop_Pa(
    rho: float,
    mu: float,
    superficial_velocity_m_s: float,
    bed_height_m: float,
    porosity: float,
    particle_diameter_m: float,
) -> float:
    eps = float(porosity)
    dp = float(particle_diameter_m)
    u = float(superficial_velocity_m_s)
    H = float(bed_height_m)
    viscous = 150.0 * mu * u * (1.0 - eps) ** 2 / (eps**3 * dp**2)
    inertial = 1.75 * rho * u**2 * (1.0 - eps) / (eps**3 * dp)
    return H * (viscous + inertial)


def gunn_nusselt_number(Re_p: float, Pr: float, porosity: float) -> float:
    """Gunn (1978) fixed/fluidised-bed particle-fluid heat-transfer correlation.

    Geometry enters explicitly through porosity and particle Reynolds number.
    This branch is retained because the Weiss reference duty is in a low-Re regime
    where the often-used Wakao high-Re fit is not a safe default.
    """
    eps = float(porosity)
    re = max(float(Re_p), 0.0)
    pr13 = max(float(Pr), 1e-12) ** (1.0 / 3.0)
    a = 7.0 - 10.0 * eps + 5.0 * eps**2
    b = 1.33 - 2.40 * eps + 1.20 * eps**2
    return a * (1.0 + 0.7 * re**0.2 * pr13) + b * re**0.7 * pr13


def packed_bed_transport(
    *,
    internal_diameter_m: float,
    bed_height_m: float,
    porosity: float,
    particle_diameter_m: float,
    mass_flow_kg_s: float,
    mean_temperature_K: float,
    cp_fluid_J_kgK: float,
) -> dict[str, float | bool | str]:
    props = solar_salt_properties(mean_temperature_K, cp_fluid_J_kgK)
    area = math.pi * internal_diameter_m**2 / 4.0
    u = mass_flow_kg_s / (props.density_kg_m3 * area)
    Re_p = props.density_kg_m3 * u * particle_diameter_m / props.viscosity_Pa_s
    Pr = props.cp_J_kgK * props.viscosity_Pa_s / props.thermal_conductivity_W_mK
    Nu = gunn_nusselt_number(Re_p, Pr, porosity)
    h = Nu * props.thermal_conductivity_W_mK / particle_diameter_m
    a_s = 6.0 * (1.0 - porosity) / particle_diameter_m
    delta_p = ergun_pressure_drop_Pa(
        props.density_kg_m3,
        props.viscosity_Pa_s,
        u,
        bed_height_m,
        porosity,
        particle_diameter_m,
    )
    # Gunn's original correlation covers fixed/fluidized beds over a broad Re range;
    # keep the domain check explicit and conservative for dense packed beds.
    correlation_valid = 0.35 <= porosity < 1.0 and Re_p <= 1.0e5 and Pr > 0
    return {
        **props.as_dict(),
        "cross_section_m2": area,
        "superficial_velocity_m_s": u,
        "particle_Re": Re_p,
        "Pr": Pr,
        "Nu_gunn": Nu,
        "interphase_h_W_m2K": h,
        "specific_interphase_area_m2_m3": a_s,
        "volumetric_interphase_hA_W_m3K": h * a_s,
        "pressure_drop_Pa": delta_p,
        "correlation_valid": correlation_valid,
        "heat_transfer_correlation": "Gunn1978",
        "pressure_drop_correlation": "Ergun1952",
    }


def _front_thickness(z: np.ndarray, T_s: np.ndarray, cold_K: float, hot_K: float) -> float:
    span = hot_K - cold_K
    if span <= 0:
        return float("nan")
    theta = np.clip((T_s - cold_K) / span, 0.0, 1.0)
    # profile normally drops from hot at inlet to cold at outlet during charging
    if np.nanmax(theta) < 0.9 or np.nanmin(theta) > 0.1:
        return float("nan")
    def crossing(target: float) -> float:
        for i in range(len(z) - 1):
            y0, y1 = theta[i], theta[i + 1]
            if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
                frac = (target - y0) / (y1 - y0)
                return float(z[i] + frac * (z[i + 1] - z[i]))
        return float("nan")
    z90 = crossing(0.9)
    z10 = crossing(0.1)
    return abs(z10 - z90) if math.isfinite(z90) and math.isfinite(z10) else float("nan")


def _simulate_packed_bed(
    *,
    internal_diameter_m: float,
    bed_height_m: float,
    porosity: float,
    particle_diameter_m: float,
    mass_flow_kg_s: float,
    cold_temperature_K: float,
    hot_temperature_K: float,
    fluid_cp_J_kgK: float,
    solid_density_kg_m3: float,
    solid_cp_J_kgK: float,
    mode: str,
    n_cells: int = 30,
    max_step_s: float = 2.0,
) -> dict[str, Any]:
    if mode not in {"charge", "discharge"}:
        raise ValueError("mode must be charge or discharge")
    if hot_temperature_K <= cold_temperature_K:
        raise ValueError("hot_temperature_K must exceed cold_temperature_K")
    mean_T = 0.5 * (cold_temperature_K + hot_temperature_K)
    tr = packed_bed_transport(
        internal_diameter_m=internal_diameter_m,
        bed_height_m=bed_height_m,
        porosity=porosity,
        particle_diameter_m=particle_diameter_m,
        mass_flow_kg_s=mass_flow_kg_s,
        mean_temperature_K=mean_T,
        cp_fluid_J_kgK=fluid_cp_J_kgK,
    )
    rho_f = float(tr["density_kg_m3"])
    cp_f = float(fluid_cp_J_kgK)
    hA = float(tr["volumetric_interphase_hA_W_m3K"])
    area = float(tr["cross_section_m2"])
    u = float(tr["superficial_velocity_m_s"])
    eps = float(porosity)
    H = float(bed_height_m)
    dz = H / n_cells
    z = (np.arange(n_cells) + 0.5) * dz
    cell_vol = area * dz
    cap_f_vol = eps * rho_f * cp_f
    cap_s_vol = (1.0 - eps) * solid_density_kg_m3 * solid_cp_J_kgK
    adv_velocity = u / eps
    tau_f = cap_f_vol / max(hA, 1e-12)
    tau_s = cap_s_vol / max(hA, 1e-12)
    cfl_step = 0.35 * dz / max(adv_velocity, 1e-12)
    stable_step = min(float(max_step_s), cfl_step, 0.20 * tau_f, 0.20 * tau_s)
    stable_step = max(stable_step, 1e-3)

    cold = float(cold_temperature_K)
    hot = float(hot_temperature_K)
    initial = cold if mode == "charge" else hot
    inlet = hot if mode == "charge" else cold
    T_f = np.full(n_cells, initial, dtype=float)
    T_s = np.full(n_cells, initial, dtype=float)

    volume = area * H
    C_total_J_K = (cap_f_vol + cap_s_vol) * volume
    Q_span_J = C_total_J_K * (hot - cold)
    nominal_s = Q_span_J / max(mass_flow_kg_s * cp_f * (hot - cold), 1e-12)
    t_end = 1.25 * nominal_s
    n_steps = max(2, int(math.ceil(t_end / stable_step)))
    dt = t_end / n_steps

    target_energy_fracs = [0.1, 0.5, 0.9]
    hit_times: dict[float, float | None] = {x: None for x in target_energy_fracs}
    front_at_50 = float("nan")
    useful_discharge_J = 0.0
    total_discharge_J = 0.0
    high_grade_threshold = cold + 0.9 * (hot - cold)
    time_s = 0.0
    outlet_trace_t: list[float] = []
    outlet_trace_T: list[float] = []

    def energy_fraction() -> float:
        E = (
            cap_f_vol * np.sum(T_f - cold) * cell_vol
            + cap_s_vol * np.sum(T_s - cold) * cell_vol
        )
        return float(np.clip(E / max(Q_span_J, 1e-12), 0.0, 1.0))

    for step in range(n_steps):
        # explicit first-order upwind advection, with direct LTNE interphase exchange
        upstream = np.empty_like(T_f)
        upstream[0] = inlet
        upstream[1:] = T_f[:-1]
        adv = -adv_velocity * (T_f - upstream) / dz
        q_fs = hA * (T_s - T_f)
        dTf = adv + q_fs / cap_f_vol
        dTs = -q_fs / cap_s_vol
        T_f = T_f + dt * dTf
        T_s = T_s + dt * dTs
        # numerical guard only; it cannot create energy beyond inlet bounds
        T_f = np.clip(T_f, cold, hot)
        T_s = np.clip(T_s, cold, hot)
        time_s += dt

        if step % max(1, n_steps // 250) == 0 or step == n_steps - 1:
            outlet_trace_t.append(time_s)
            outlet_trace_T.append(float(T_f[-1]))

        ef = energy_fraction()
        if mode == "charge":
            for target in target_energy_fracs:
                if hit_times[target] is None and ef >= target:
                    hit_times[target] = time_s
            if hit_times[0.5] is not None and not math.isfinite(front_at_50):
                front_at_50 = _front_thickness(z, T_s, cold, hot)
        else:
            Tout = float(T_f[-1])
            power = mass_flow_kg_s * cp_f * max(Tout - cold, 0.0)
            total_discharge_J += power * dt
            if Tout >= high_grade_threshold:
                useful_discharge_J += power * dt

    result = {
        "mode": mode,
        "n_cells": n_cells,
        "time_step_s": dt,
        "nominal_energy_time_s": nominal_s,
        "simulated_time_s": time_s,
        "outlet_trace_time_s": outlet_trace_t,
        "outlet_trace_temperature_K": outlet_trace_T,
        "final_outlet_temperature_K": float(T_f[-1]),
        "final_fluid_profile_K": T_f.tolist(),
        "final_solid_profile_K": T_s.tolist(),
        "z_m": z.tolist(),
        "transport": tr,
    }
    if mode == "charge":
        result.update({
            "t10_energy_s": hit_times[0.1],
            "t50_energy_s": hit_times[0.5],
            "t90_energy_s": hit_times[0.9],
            "thermal_front_thickness_at_50pct_m": front_at_50,
            "thermal_front_thickness_ratio": front_at_50 / H if math.isfinite(front_at_50) else None,
        })
    else:
        result.update({
            "total_extracted_energy_MWh": total_discharge_J / J_PER_MWH,
            "high_grade_extracted_energy_MWh": useful_discharge_J / J_PER_MWH,
            "high_grade_fraction_of_initial": useful_discharge_J / max(Q_span_J, 1e-12),
        })
    return result


def evaluate_sensible_packed_bed(
    *,
    design_id: str,
    internal_diameter_m: float,
    bed_height_m: float,
    porosity: float,
    particle_diameter_m: float,
    mass_flow_kg_s: float,
    cold_temperature_K: float,
    hot_temperature_K: float,
    fluid_cp_J_kgK: float,
    solid_density_kg_m3: float,
    solid_cp_J_kgK: float,
    n_cells: int = 30,
) -> dict[str, Any]:
    charge = _simulate_packed_bed(
        internal_diameter_m=internal_diameter_m,
        bed_height_m=bed_height_m,
        porosity=porosity,
        particle_diameter_m=particle_diameter_m,
        mass_flow_kg_s=mass_flow_kg_s,
        cold_temperature_K=cold_temperature_K,
        hot_temperature_K=hot_temperature_K,
        fluid_cp_J_kgK=fluid_cp_J_kgK,
        solid_density_kg_m3=solid_density_kg_m3,
        solid_cp_J_kgK=solid_cp_J_kgK,
        mode="charge",
        n_cells=n_cells,
    )
    discharge = _simulate_packed_bed(
        internal_diameter_m=internal_diameter_m,
        bed_height_m=bed_height_m,
        porosity=porosity,
        particle_diameter_m=particle_diameter_m,
        mass_flow_kg_s=mass_flow_kg_s,
        cold_temperature_K=cold_temperature_K,
        hot_temperature_K=hot_temperature_K,
        fluid_cp_J_kgK=fluid_cp_J_kgK,
        solid_density_kg_m3=solid_density_kg_m3,
        solid_cp_J_kgK=solid_cp_J_kgK,
        mode="discharge",
        n_cells=n_cells,
    )
    tr = charge["transport"]
    area = float(tr["cross_section_m2"])
    volume = area * bed_height_m
    rho_f = float(tr["density_kg_m3"])
    mean_T = 0.5 * (cold_temperature_K + hot_temperature_K)
    delta_T = hot_temperature_K - cold_temperature_K
    fluid_mass = porosity * volume * rho_f
    solid_mass = (1.0 - porosity) * volume * solid_density_kg_m3
    capacity_MWh = (fluid_mass * fluid_cp_J_kgK + solid_mass * solid_cp_J_kgK) * delta_T / J_PER_MWH
    t90 = charge["t90_energy_s"]
    pump_power_W = float(tr["pressure_drop_Pa"]) * (mass_flow_kg_s / rho_f)
    pump_fraction = None
    if t90 is not None and capacity_MWh > 0:
        pump_fraction = pump_power_W * float(t90) / (capacity_MWh * J_PER_MWH)
    return {
        "design_id": design_id,
        "architecture": "sensible_packed_bed_LTNE",
        "evidence_status": "runnable_base_solar_salt_transport_only",
        "internal_diameter_m": internal_diameter_m,
        "bed_height_m": bed_height_m,
        "porosity": porosity,
        "particle_diameter_m": particle_diameter_m,
        "mass_flow_kg_s": mass_flow_kg_s,
        "cold_temperature_K": cold_temperature_K,
        "hot_temperature_K": hot_temperature_K,
        "fluid_cp_J_kgK": fluid_cp_J_kgK,
        "storage_volume_m3": volume,
        "stored_heat_MWh": capacity_MWh,
        "volumetric_capacity_MWh_m3": capacity_MWh / volume,
        "charge_t90_h": None if t90 is None else float(t90) / 3600.0,
        "charge_t50_h": None if charge["t50_energy_s"] is None else float(charge["t50_energy_s"]) / 3600.0,
        "thermal_front_thickness_ratio": charge["thermal_front_thickness_ratio"],
        "pressure_drop_kPa": float(tr["pressure_drop_Pa"]) / 1000.0,
        "pump_power_W": pump_power_W,
        "pump_energy_fraction": pump_fraction,
        "high_grade_discharge_fraction": discharge["high_grade_fraction_of_initial"],
        "particle_Re": float(tr["particle_Re"]),
        "Pr": float(tr["Pr"]),
        "Nu": float(tr["Nu_gunn"]),
        "correlation_valid": bool(tr["correlation_valid"]),
        "transport": tr,
        "charge_trace": {
            "time_s": charge["outlet_trace_time_s"],
            "outlet_temperature_K": charge["outlet_trace_temperature_K"],
        },
        "discharge_trace": {
            "time_s": discharge["outlet_trace_time_s"],
            "outlet_temperature_K": discharge["outlet_trace_temperature_K"],
        },
    }


def array_scale(unit: dict[str, Any], n_parallel: int, n_series: int) -> dict[str, float | int]:
    if n_parallel < 1 or n_series < 1:
        raise ValueError("array counts must be positive")
    total_units = n_parallel * n_series
    rho = float(unit["transport"]["density_kg_m3"])
    unit_flow = float(unit["mass_flow_kg_s"])
    unit_dp_Pa = float(unit["pressure_drop_kPa"]) * 1000.0
    total_flow = unit_flow * n_parallel
    total_dp = unit_dp_Pa * n_series
    pump_W = total_dp * total_flow / rho
    return {
        "n_parallel": n_parallel,
        "n_series": n_series,
        "n_units": total_units,
        "array_stored_heat_MWh": float(unit["stored_heat_MWh"]) * total_units,
        "array_flow_kg_s": total_flow,
        "array_pressure_drop_kPa": total_dp / 1000.0,
        "array_pump_power_W": pump_W,
    }


def pareto_front(records: Iterable[dict[str, Any]], objective_keys: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = list(records)
    if not rows:
        return []
    vals = np.asarray([[float(r[k]) for k, _ in objective_keys] for r in rows], dtype=float)
    for j, (_, sense) in enumerate(objective_keys):
        if sense == "maximize":
            vals[:, j] *= -1.0
        elif sense != "minimize":
            raise ValueError(f"unknown sense {sense}")
    keep = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        if not keep[i]:
            continue
        dominates_i = np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return [r for i, r in enumerate(rows) if keep[i]]

# ---- JIT-accelerated scalar-output kernel for large design screens ----
try:
    from numba import njit
except Exception:  # pragma: no cover - pure Python fallback
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


@njit(cache=True)
def _fast_core(
    mode_charge: int,
    n_cells: int,
    n_steps: int,
    dt: float,
    dz: float,
    adv_velocity: float,
    cap_f_vol: float,
    cap_s_vol: float,
    hA: float,
    cell_vol: float,
    Q_span_J: float,
    inlet: float,
    initial: float,
    cold: float,
    hot: float,
    mass_flow: float,
    cp_f: float,
):
    T_f = np.empty(n_cells, dtype=np.float64)
    T_s = np.empty(n_cells, dtype=np.float64)
    for i in range(n_cells):
        T_f[i] = initial
        T_s[i] = initial
    t10 = -1.0; t50 = -1.0; t90 = -1.0
    front50 = -1.0
    useful_J = 0.0; total_J = 0.0; inventory_J = 0.0
    high_grade_duration_s = 0.0
    high_grade = cold + 0.9 * (hot - cold)
    time_s = 0.0
    upstream = np.empty(n_cells, dtype=np.float64)
    dTf = np.empty(n_cells, dtype=np.float64)
    dTs = np.empty(n_cells, dtype=np.float64)
    for _ in range(n_steps):
        upstream[0] = inlet
        for i in range(1, n_cells):
            upstream[i] = T_f[i-1]
        for i in range(n_cells):
            adv = -adv_velocity * (T_f[i] - upstream[i]) / dz
            qfs = hA * (T_s[i] - T_f[i])
            dTf[i] = adv + qfs / cap_f_vol
            dTs[i] = -qfs / cap_s_vol
        for i in range(n_cells):
            tf = T_f[i] + dt*dTf[i]
            ts = T_s[i] + dt*dTs[i]
            if tf < cold: tf = cold
            elif tf > hot: tf = hot
            if ts < cold: ts = cold
            elif ts > hot: ts = hot
            T_f[i] = tf; T_s[i] = ts
        time_s += dt
        if mode_charge == 1:
            E = 0.0
            for i in range(n_cells):
                E += (cap_f_vol*(T_f[i]-cold) + cap_s_vol*(T_s[i]-cold))*cell_vol
            inventory_J = E
            # Boundary enthalpy input is tracked independently from the state
            # inventory so clipping/numerical diffusion is visible in QA.
            total_J += mass_flow*cp_f*max(inlet-T_f[n_cells-1],0.0)*dt
            ef = E / Q_span_J
            if t10 < 0.0 and ef >= 0.1: t10 = time_s
            if t50 < 0.0 and ef >= 0.5:
                t50 = time_s
                # direct grid estimate of 10-90% front thickness
                z90 = -1.0; z10 = -1.0
                span = hot-cold
                for i in range(n_cells-1):
                    th0=(T_s[i]-cold)/span; th1=(T_s[i+1]-cold)/span
                    if z90 < 0.0 and (th0-0.9)*(th1-0.9) <= 0.0 and th0 != th1:
                        z90=(i+0.5)*dz + (0.9-th0)/(th1-th0)*dz
                    if z10 < 0.0 and (th0-0.1)*(th1-0.1) <= 0.0 and th0 != th1:
                        z10=(i+0.5)*dz + (0.1-th0)/(th1-th0)*dz
                if z90 >= 0.0 and z10 >= 0.0:
                    front50 = abs(z10-z90)
            if t90 < 0.0 and ef >= 0.9: t90 = time_s
        else:
            Tout = T_f[n_cells-1]
            power = mass_flow*cp_f*max(Tout-cold,0.0)
            total_J += power*dt
            if Tout >= high_grade:
                useful_J += power*dt
                high_grade_duration_s = time_s
            inventory_J = 0.0
            for i in range(n_cells):
                inventory_J += (cap_f_vol*(T_f[i]-cold) + cap_s_vol*(T_s[i]-cold))*cell_vol
            extracted_fraction = 1.0 - inventory_J/Q_span_J
            if t10 < 0.0 and extracted_fraction >= 0.1: t10 = time_s
            if t50 < 0.0 and extracted_fraction >= 0.5: t50 = time_s
            if t90 < 0.0 and extracted_fraction >= 0.9: t90 = time_s
    return t10,t50,t90,front50,useful_J,total_J,T_f[n_cells-1],inventory_J,high_grade_duration_s


def evaluate_sensible_packed_bed_fast(
    *,
    design_id: str,
    internal_diameter_m: float,
    bed_height_m: float,
    porosity: float,
    particle_diameter_m: float,
    mass_flow_kg_s: float,
    cold_temperature_K: float,
    hot_temperature_K: float,
    fluid_cp_J_kgK: float,
    solid_density_kg_m3: float,
    solid_cp_J_kgK: float,
    n_cells: int = 16,
) -> dict[str, Any]:
    mean_T=0.5*(cold_temperature_K+hot_temperature_K)
    tr=packed_bed_transport(
        internal_diameter_m=internal_diameter_m, bed_height_m=bed_height_m,
        porosity=porosity, particle_diameter_m=particle_diameter_m,
        mass_flow_kg_s=mass_flow_kg_s, mean_temperature_K=mean_T,
        cp_fluid_J_kgK=fluid_cp_J_kgK)
    rho_f=float(tr['density_kg_m3']); cp_f=float(fluid_cp_J_kgK); hA=float(tr['volumetric_interphase_hA_W_m3K'])
    area=float(tr['cross_section_m2']); u=float(tr['superficial_velocity_m_s']); eps=float(porosity); H=float(bed_height_m)
    dz=H/n_cells; cell_vol=area*dz; cap_f_vol=eps*rho_f*cp_f; cap_s_vol=(1-eps)*solid_density_kg_m3*solid_cp_J_kgK
    adv_velocity=u/eps; tau_f=cap_f_vol/max(hA,1e-12); tau_s=cap_s_vol/max(hA,1e-12)
    stable=min(2.0,0.35*dz/max(adv_velocity,1e-12),0.20*tau_f,0.20*tau_s); stable=max(stable,1e-3)
    cold=float(cold_temperature_K); hot=float(hot_temperature_K); delta=hot-cold; V=area*H
    Q_span=(cap_f_vol+cap_s_vol)*V*delta
    nominal=Q_span/max(mass_flow_kg_s*cp_f*delta,1e-12); t_end=1.25*nominal
    n_steps=max(2,int(math.ceil(t_end/stable))); dt=t_end/n_steps
    ch=_fast_core(1,n_cells,n_steps,dt,dz,adv_velocity,cap_f_vol,cap_s_vol,hA,cell_vol,Q_span,hot,cold,cold,hot,mass_flow_kg_s,cp_f)
    dis=_fast_core(0,n_cells,n_steps,dt,dz,adv_velocity,cap_f_vol,cap_s_vol,hA,cell_vol,Q_span,cold,hot,cold,hot,mass_flow_kg_s,cp_f)
    fluid_mass=eps*V*rho_f; solid_mass=(1-eps)*V*solid_density_kg_m3
    capacity=(fluid_mass*cp_f+solid_mass*solid_cp_J_kgK)*delta/J_PER_MWH
    t90=None if ch[2] < 0 else float(ch[2])
    front=None if ch[3] < 0 else float(ch[3])/H
    pump_W=float(tr['pressure_drop_Pa'])*(mass_flow_kg_s/rho_f)
    pump_fraction=None if t90 is None else pump_W*t90/(capacity*J_PER_MWH)
    charge_balance_error=100.0*(float(ch[7])/max(float(ch[5]),1e-12)-1.0)
    discharge_balance_error=100.0*((float(dis[5])+float(dis[7]))/max(Q_span,1e-12)-1.0)
    return {
        'design_id':design_id,'architecture':'sensible_packed_bed_LTNE','evidence_status':'runnable_base_solar_salt_transport_only',
        'internal_diameter_m':internal_diameter_m,'bed_height_m':bed_height_m,'porosity':porosity,'particle_diameter_m':particle_diameter_m,
        'mass_flow_kg_s':mass_flow_kg_s,'cold_temperature_K':cold_temperature_K,'hot_temperature_K':hot_temperature_K,'fluid_cp_J_kgK':fluid_cp_J_kgK,
        'storage_volume_m3':V,'stored_heat_MWh':capacity,'volumetric_capacity_MWh_m3':capacity/V,
        'charge_t90_h':None if t90 is None else t90/3600.0,'charge_t50_h':None if ch[1]<0 else float(ch[1])/3600.0,
        'thermal_front_thickness_ratio':front,'pressure_drop_kPa':float(tr['pressure_drop_Pa'])/1000.0,
        'pump_power_W':pump_W,'pump_energy_fraction':pump_fraction,
        'high_grade_discharge_fraction':float(dis[4]/max(Q_span,1e-12)),
        'charge_energy_balance_error_pct':charge_balance_error,
        'discharge_energy_balance_error_pct':discharge_balance_error,
        'discharge_energy_recovery_fraction':float(dis[5]/max(Q_span,1e-12)),
        'particle_Re':float(tr['particle_Re']),'Pr':float(tr['Pr']),'Nu':float(tr['Nu_gunn']),
        'correlation_valid':bool(tr['correlation_valid']),
    }

# ---- Architecture kernels: topology variables enter the governing geometry/transport ----
def encapsulated_pcm_packed_bed_metrics(
    *,
    bed_volume_m3: float,
    bed_porosity: float,
    capsule_outer_diameter_m: float,
    shell_thickness_m: float,
    shell_thermal_conductivity_W_mK: float,
    external_h_W_m2K: float,
    pcm_density_kg_m3: float,
    pcm_cp_solid_J_kgK: float,
    pcm_cp_liquid_J_kgK: float,
    pcm_latent_heat_J_kg: float,
    pcm_melting_temperature_K: float,
    cold_temperature_K: float,
    hot_temperature_K: float,
    shell_density_kg_m3: float | None = None,
    shell_cp_J_kgK: float | None = None,
) -> dict[str, float]:
    """Evidence-gated reduced-order kernel for spherical macro-PCM capsules.

    Capsule diameter and shell thickness alter PCM displacement, shell mass,
    heat-transfer area and spherical shell resistance directly. This function
    intentionally requires material properties instead of applying a literature
    melting-time percentage as a correction factor.
    """
    if not (0 < bed_porosity < 1):
        raise ValueError("bed_porosity must lie in (0,1)")
    ro = 0.5 * capsule_outer_diameter_m
    ri = ro - shell_thickness_m
    if ri <= 0 or shell_thermal_conductivity_W_mK <= 0 or external_h_W_m2K <= 0:
        raise ValueError("non-physical capsule geometry/transport input")
    if not (cold_temperature_K < pcm_melting_temperature_K < hot_temperature_K):
        raise ValueError("melting temperature must lie inside the operating span")
    v_outer = 4.0 / 3.0 * math.pi * ro**3
    v_pcm = 4.0 / 3.0 * math.pi * ri**3
    v_shell = v_outer - v_pcm
    n_caps = (1.0 - bed_porosity) * bed_volume_m3 / v_outer
    m_pcm = n_caps * v_pcm * pcm_density_kg_m3
    q_pcm_J = m_pcm * (
        pcm_cp_solid_J_kgK * (pcm_melting_temperature_K - cold_temperature_K)
        + pcm_latent_heat_J_kg
        + pcm_cp_liquid_J_kgK * (hot_temperature_K - pcm_melting_temperature_K)
    )
    q_shell_J = 0.0
    if shell_density_kg_m3 is not None and shell_cp_J_kgK is not None:
        q_shell_J = n_caps * v_shell * shell_density_kg_m3 * shell_cp_J_kgK * (hot_temperature_K - cold_temperature_K)
    area_one = 4.0 * math.pi * ro**2
    area_total = n_caps * area_one
    r_conv_one = 1.0 / (external_h_W_m2K * area_one)
    r_shell_one = (1.0 / (4.0 * math.pi * shell_thermal_conductivity_W_mK)) * (1.0 / ri - 1.0 / ro)
    ua_total = n_caps / (r_conv_one + r_shell_one)
    return {
        "capsule_count": n_caps,
        "pcm_volume_fraction_of_bed": n_caps * v_pcm / bed_volume_m3,
        "shell_volume_fraction_of_bed": n_caps * v_shell / bed_volume_m3,
        "pcm_mass_kg": m_pcm,
        "external_heat_transfer_area_m2": area_total,
        "single_capsule_convective_resistance_K_W": r_conv_one,
        "single_capsule_shell_resistance_K_W": r_shell_one,
        "aggregate_UA_W_K": ua_total,
        "stored_heat_MWh": (q_pcm_J + q_shell_J) / J_PER_MWH,
        "volumetric_capacity_MWh_m3": (q_pcm_J + q_shell_J) / J_PER_MWH / bed_volume_m3,
    }


def finned_pcm_geometry_metrics(
    *,
    envelope_volume_m3: float,
    base_heat_transfer_area_m2: float,
    fin_count: int,
    fin_length_m: float,
    fin_width_m: float,
    fin_thickness_m: float,
    fin_thermal_conductivity_W_mK: float,
    local_pcm_h_W_m2K: float,
    pcm_density_kg_m3: float,
    pcm_energy_density_J_kg: float,
    fin_density_kg_m3: float | None = None,
    fin_cp_J_kgK: float | None = None,
    operating_delta_T_K: float | None = None,
) -> dict[str, float]:
    """Rectangular-fin reduced-order geometry kernel.

    Fin number/length/thickness directly trade effective heat-transfer area
    against displaced PCM volume. Angle/orientation effects on natural convection
    are deliberately *not* collapsed into an empirical multiplier; they require
    a matched experimental calibration or independently validated reduced-order branch.
    """
    if fin_count < 0 or min(envelope_volume_m3, base_heat_transfer_area_m2, fin_length_m, fin_width_m, fin_thickness_m,
                            fin_thermal_conductivity_W_mK, local_pcm_h_W_m2K, pcm_density_kg_m3, pcm_energy_density_J_kg) <= 0:
        raise ValueError("non-physical fin/PCM input")
    fin_volume = fin_count * fin_length_m * fin_width_m * fin_thickness_m
    if fin_volume >= envelope_volume_m3:
        raise ValueError("fins displace all PCM volume")
    pcm_volume = envelope_volume_m3 - fin_volume
    # Thin straight rectangular fin, adiabatic-tip efficiency.
    m = math.sqrt(2.0 * local_pcm_h_W_m2K / (fin_thermal_conductivity_W_mK * fin_thickness_m))
    mL = m * fin_length_m
    eta = 1.0 if mL < 1e-12 else math.tanh(mL) / mL
    fin_area_geometric = fin_count * (2.0 * fin_length_m * fin_width_m + fin_width_m * fin_thickness_m)
    effective_area = base_heat_transfer_area_m2 + eta * fin_area_geometric
    q_pcm = pcm_volume * pcm_density_kg_m3 * pcm_energy_density_J_kg
    q_fin = 0.0
    if fin_density_kg_m3 is not None and fin_cp_J_kgK is not None and operating_delta_T_K is not None:
        q_fin = fin_volume * fin_density_kg_m3 * fin_cp_J_kgK * operating_delta_T_K
    return {
        "fin_volume_m3": fin_volume,
        "pcm_volume_m3": pcm_volume,
        "pcm_displacement_fraction": fin_volume / envelope_volume_m3,
        "fin_efficiency": eta,
        "geometric_fin_area_m2": fin_area_geometric,
        "effective_heat_transfer_area_m2": effective_area,
        "effective_UA_W_K": local_pcm_h_W_m2K * effective_area,
        "stored_heat_MWh": (q_pcm + q_fin) / J_PER_MWH,
        "volumetric_capacity_MWh_m3": (q_pcm + q_fin) / J_PER_MWH / envelope_volume_m3,
    }


def foam_pcm_metrics(
    *,
    envelope_volume_m3: float,
    foam_porosity: float,
    foam_solid_density_kg_m3: float,
    foam_solid_cp_J_kgK: float,
    pcm_density_kg_m3: float,
    pcm_energy_density_J_kg: float,
    permeability_m2: float,
    forchheimer_coefficient: float,
    flow_length_m: float,
    flow_area_m2: float,
    mass_flow_kg_s: float,
    fluid_density_kg_m3: float,
    fluid_viscosity_Pa_s: float,
    operating_delta_T_K: float,
) -> dict[str, float]:
    """Open-cell foam capacity + Darcy-Forchheimer hydraulic kernel.

    Permeability and inertial coefficient must come from matched foam evidence or
    resolved geometry; PPI/porosity outcome percentages are not used as multipliers.
    """
    if not (0 < foam_porosity < 1) or min(permeability_m2, flow_length_m, flow_area_m2, fluid_density_kg_m3, fluid_viscosity_Pa_s) <= 0:
        raise ValueError("non-physical foam input")
    pcm_vol = envelope_volume_m3 * foam_porosity
    solid_vol = envelope_volume_m3 * (1.0 - foam_porosity)
    q_pcm = pcm_vol * pcm_density_kg_m3 * pcm_energy_density_J_kg
    q_foam = solid_vol * foam_solid_density_kg_m3 * foam_solid_cp_J_kgK * operating_delta_T_K
    u = mass_flow_kg_s / (fluid_density_kg_m3 * flow_area_m2)
    dp = flow_length_m * (
        fluid_viscosity_Pa_s * u / permeability_m2
        + fluid_density_kg_m3 * forchheimer_coefficient * u * u / math.sqrt(permeability_m2)
    )
    return {
        "pcm_volume_m3": pcm_vol,
        "foam_solid_volume_m3": solid_vol,
        "stored_heat_MWh": (q_pcm + q_foam) / J_PER_MWH,
        "volumetric_capacity_MWh_m3": (q_pcm + q_foam) / J_PER_MWH / envelope_volume_m3,
        "superficial_velocity_m_s": u,
        "pressure_drop_kPa": dp / 1000.0,
    }


def structured_square_channel_metrics(
    *,
    channel_count: int,
    channel_width_m: float,
    wall_thickness_m: float,
    channel_length_m: float,
    mass_flow_kg_s: float,
    fluid_density_kg_m3: float,
    fluid_viscosity_Pa_s: float,
    fluid_cp_J_kgK: float,
    fluid_thermal_conductivity_W_mK: float,
    solid_density_kg_m3: float,
    solid_cp_J_kgK: float,
    operating_delta_T_K: float,
) -> dict[str, float | str]:
    """Square honeycomb-channel hydraulic/heat-transfer geometry kernel.

    Uses fully developed laminar Nu=3.61 for square ducts; for turbulent flow a
    Gnielinski-type relation is used. The branch is generic and must be validated
    against a matched honeycomb experiment before cross-architecture ranking.
    """
    if channel_count < 1 or min(channel_width_m, wall_thickness_m, channel_length_m, mass_flow_kg_s,
                                fluid_density_kg_m3, fluid_viscosity_Pa_s, fluid_cp_J_kgK,
                                fluid_thermal_conductivity_W_mK, solid_density_kg_m3, solid_cp_J_kgK) <= 0:
        raise ValueError("non-physical structured-channel input")
    a = channel_width_m
    pitch = a + wall_thickness_m
    flow_area = channel_count * a * a
    frontal_area = channel_count * pitch * pitch
    solid_area = frontal_area - flow_area
    solid_volume = solid_area * channel_length_m
    envelope_volume = frontal_area * channel_length_m
    velocity = mass_flow_kg_s / (fluid_density_kg_m3 * flow_area)
    Dh = a
    Re = fluid_density_kg_m3 * velocity * Dh / fluid_viscosity_Pa_s
    Pr = fluid_cp_J_kgK * fluid_viscosity_Pa_s / fluid_thermal_conductivity_W_mK
    if Re < 2300.0:
        f = 56.91 / Re  # Darcy friction factor for a square duct
        Nu = 3.61
        regime = "laminar_square_duct"
    else:
        f = (0.79 * math.log(Re) - 1.64) ** -2
        Nu = (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
        regime = "turbulent_gnielinski"
    dp = f * channel_length_m / Dh * 0.5 * fluid_density_kg_m3 * velocity**2
    h = Nu * fluid_thermal_conductivity_W_mK / Dh
    wetted_area = channel_count * 4.0 * a * channel_length_m
    q_solid = solid_volume * solid_density_kg_m3 * solid_cp_J_kgK * operating_delta_T_K
    return {
        "envelope_volume_m3": envelope_volume,
        "solid_volume_m3": solid_volume,
        "open_area_fraction": flow_area / frontal_area,
        "hydraulic_diameter_m": Dh,
        "mean_velocity_m_s": velocity,
        "Re": Re,
        "Pr": Pr,
        "friction_factor": f,
        "Nu": Nu,
        "convective_h_W_m2K": h,
        "wetted_heat_transfer_area_m2": wetted_area,
        "UA_W_K": h * wetted_area,
        "pressure_drop_kPa": dp / 1000.0,
        "solid_sensible_capacity_MWh": q_solid / J_PER_MWH,
        "regime": regime,
    }


def size_parallel_modular_system(
    *,
    unit_geometry: dict[str, float],
    target_thermal_power_MW: float,
    storage_duration_h: float,
    max_unit_mass_flow_kg_s: float,
    fluid_cp_J_kgK: float,
    cold_temperature_K: float,
    hot_temperature_K: float,
    solid_density_kg_m3: float,
    solid_cp_J_kgK: float,
    n_cells: int = 16,
) -> dict[str, float | int | str]:
    """Size a parallel modular array from thermal-power and energy requirements.

    The total duty sets mass flow, while duration sets capacity. Per-unit flow is
    then fed back into Ergun/Gunn/LTNE rather than keeping the reference pressure
    drop fixed. Manifold losses and series thermal coupling are outside this branch.
    """
    if target_thermal_power_MW <= 0 or storage_duration_h <= 0 or max_unit_mass_flow_kg_s <= 0:
        raise ValueError("system duty inputs must be positive")
    delta_T = hot_temperature_K - cold_temperature_K
    if delta_T <= 0:
        raise ValueError("hot temperature must exceed cold temperature")
    # Capacity does not depend on flow in the sensible kernel, so evaluate once.
    nominal = evaluate_sensible_packed_bed_fast(
        design_id="SYSTEM-SIZING-NOMINAL", mass_flow_kg_s=max_unit_mass_flow_kg_s,
        cold_temperature_K=cold_temperature_K, hot_temperature_K=hot_temperature_K,
        fluid_cp_J_kgK=fluid_cp_J_kgK, solid_density_kg_m3=solid_density_kg_m3,
        solid_cp_J_kgK=solid_cp_J_kgK, n_cells=n_cells, **unit_geometry)
    unit_capacity = float(nominal["stored_heat_MWh"])
    required_energy = target_thermal_power_MW * storage_duration_h
    total_flow = target_thermal_power_MW * 1e6 / (fluid_cp_J_kgK * delta_T)
    n_capacity = int(math.ceil(required_energy / unit_capacity))
    n_flow = int(math.ceil(total_flow / max_unit_mass_flow_kg_s))
    n_parallel = max(1, n_capacity, n_flow)
    unit_flow = total_flow / n_parallel
    resolved = evaluate_sensible_packed_bed_fast(
        design_id="SYSTEM-SIZING-RESOLVED", mass_flow_kg_s=unit_flow,
        cold_temperature_K=cold_temperature_K, hot_temperature_K=hot_temperature_K,
        fluid_cp_J_kgK=fluid_cp_J_kgK, solid_density_kg_m3=solid_density_kg_m3,
        solid_cp_J_kgK=solid_cp_J_kgK, n_cells=n_cells, **unit_geometry)
    rho = float(solar_salt_properties(0.5 * (cold_temperature_K + hot_temperature_K), fluid_cp_J_kgK).density_kg_m3)
    total_dp_Pa = float(resolved["pressure_drop_kPa"]) * 1000.0
    pump_power = total_dp_Pa * (total_flow / rho)
    unit_volume = math.pi * float(unit_geometry["internal_diameter_m"])**2 / 4.0 * float(unit_geometry["bed_height_m"])
    installed_capacity = n_parallel * float(resolved["stored_heat_MWh"])
    charge_t90 = float(resolved["charge_t90_h"])
    return {
        "target_thermal_power_MW": target_thermal_power_MW,
        "storage_duration_h": storage_duration_h,
        "required_energy_MWh": required_energy,
        "required_total_mass_flow_kg_s": total_flow,
        "parallel_unit_count": n_parallel,
        "capacity_limited_unit_count": n_capacity,
        "flow_limited_unit_count": n_flow,
        "resolved_unit_mass_flow_kg_s": unit_flow,
        "unit_packed_zone_volume_m3": unit_volume,
        "installed_packed_zone_volume_m3": n_parallel * unit_volume,
        "installed_packed_zone_capacity_MWh": installed_capacity,
        "capacity_oversize_pct": (installed_capacity / required_energy - 1.0) * 100.0,
        "resolved_unit_charge_t90_h": charge_t90,
        "charge_t90_to_duty_duration_ratio": charge_t90 / storage_duration_h,
        "system_pressure_drop_kPa_excluding_manifolds": float(resolved["pressure_drop_kPa"]),
        "system_pump_power_W_excluding_manifolds": pump_power,
        "model_scope": "parallel modular packed-zone sizing; manifolds/headers and balance-of-plant excluded",
    }
