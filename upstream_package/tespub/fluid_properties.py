from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .crossscale_dynamic import J_PER_MWH, _fast_core, ergun_pressure_drop_Pa, gunn_nusselt_number


@dataclass(frozen=True)
class FluidModelSpec:
    model_id: str
    material_name: str
    family: str
    composition_basis: str
    composition: str
    valid_temperature_min_C: float
    valid_temperature_max_C: float
    quantitative_use_class: str
    source_ids: tuple[str, ...]
    evidence_note: str


@dataclass(frozen=True)
class FluidProperties:
    model_id: str
    temperature_C: float
    density_kg_m3: float
    cp_J_kgK: float
    viscosity_Pa_s: float
    thermal_conductivity_W_mK: float
    source_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["source_ids"] = list(self.source_ids)
        return out


# This frozen executable registry keeps only source-complete high-temperature property models.
# Broader salts remain in the evidence database but cannot be evaluated here.
_MODEL_SPECS: dict[str, FluidModelSpec] = {
    "SOLAR_SALT_60_40_V10": FluidModelSpec(
        model_id="SOLAR_SALT_60_40_V10",
        material_name="Solar Salt (NaNO3-KNO3 60:40 wt%)",
        family="nitrate",
        composition_basis="wt%",
        composition="NaNO3 60; KNO3 40",
        valid_temperature_min_C=300.0,
        # The controlled comparison uses 565 C as its ceiling even though the
        # legacy transport fits extend to 600 C, matching the registered
        # commercial/design-basis hot-side operating limit.
        valid_temperature_max_C=565.0,
        quantitative_use_class="Q1_authoritative_design_basis_full_property_vector",
        source_ids=("ZAVOICO_SANDIA_SOLAR_SALT_2001",),
        evidence_note="Density, Cp, viscosity and thermal conductivity use the internally consistent Sandia Solar-Salt design-basis correlations. The lower-temperature Andreu DSC states are retained only in the nanoparticle evidence layer and are not extrapolated into this duty.",
    ),
    "NREL_MGCL2_KCL_NACL_BASELINE_V10": FluidModelSpec(
        model_id="NREL_MGCL2_KCL_NACL_BASELINE_V10",
        material_name="Purified MgCl2-KCl-NaCl baseline (NREL)",
        family="chloride",
        composition_basis="mol%",
        composition="MgCl2 37.51; KCl 40.92; NaCl 21.57",
        valid_temperature_min_C=475.0,
        valid_temperature_max_C=625.0,
        quantitative_use_class="Q1_official_lab_measured_correlations",
        source_ids=("NREL_ZHAO_CHLORIDE_2020",),
        evidence_note="Common domain is limited by the reported Cp correlation (475-625 C); density, viscosity and thermal conductivity have broader reported domains.",
    ),
    "AIP_FLINAK_2017_V10": FluidModelSpec(
        model_id="AIP_FLINAK_2017_V10",
        material_name="FLiNaK (LiF-NaF-KF)",
        family="fluoride",
        composition_basis="wt%",
        composition="LiF 29.21; NaF 11.69; KF 59.10",
        # Table 3 in the primary paper reports a complete rho/Cp/k set from
        # 500-600 C for the common benchmark. Restricting the executable model
        # to that overlap avoids extrapolating the fitted equations.
        valid_temperature_min_C=500.0,
        valid_temperature_max_C=600.0,
        quantitative_use_class="Q1_primary_experiment_common_table_domain",
        source_ids=("AIP_AN_CHENG_SU_ZHANG_2017", "TASIDOU_ETAL_REFERENCE_VISCOSITY_2019"),
        evidence_note="Density/Cp/k from the 2017 exact-composition primary experiment; viscosity uses the 2019 critically evaluated reference correlation (2.9% uncertainty at 95% confidence). Executable domain remains the conservative 500-600 C common overlap.",
    ),
    "AIP_LINAK_CARBONATE_2017_V10": FluidModelSpec(
        model_id="AIP_LINAK_CARBONATE_2017_V10",
        material_name="Li2CO3-Na2CO3-K2CO3 eutectic",
        family="carbonate",
        composition_basis="wt%",
        composition="Li2CO3 32.12; Na2CO3 33.36; K2CO3 34.52",
        valid_temperature_min_C=500.0,
        valid_temperature_max_C=600.0,
        quantitative_use_class="Q1_primary_experiment_common_table_domain",
        source_ids=("AIP_AN_CHENG_SU_ZHANG_2017", "TASIDOU_ETAL_REFERENCE_VISCOSITY_2019"),
        evidence_note="Density/Cp/k from the 2017 exact-composition primary experiment; viscosity uses the 2019 critically evaluated reference correlation (3.0% uncertainty at 95% confidence). Executable domain remains the conservative 500-600 C common overlap.",
    ),
    "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10": FluidModelSpec(
        model_id="AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10",
        material_name="LiF-Na2CO3-K2CO3 reciprocal salt",
        family="fluoride_carbonate",
        composition_basis="wt%",
        composition="LiF 54.20; Na2CO3 28.11; K2CO3 17.69",
        valid_temperature_min_C=500.0,
        valid_temperature_max_C=600.0,
        quantitative_use_class="Q1_primary_experiment_direct_table1_resolution",
        source_ids=("AIP_AN_CHENG_SU_ZHANG_2017",),
        evidence_note="Composition is resolved directly from Table 1 of the primary AIP article. Dynamic use remains limited to the conservative 500-600 C common tabulated property domain.",
    ),
}


_PROPERTY_SOURCE_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "SOLAR_SALT_60_40_V10": {
        "density": ("ZAVOICO_SANDIA_SOLAR_SALT_2001",),
        "specific_heat_capacity": ("ZAVOICO_SANDIA_SOLAR_SALT_2001",),
        "dynamic_viscosity": ("ZAVOICO_SANDIA_SOLAR_SALT_2001",),
        "thermal_conductivity": ("ZAVOICO_SANDIA_SOLAR_SALT_2001",),
    },
    "NREL_MGCL2_KCL_NACL_BASELINE_V10": {p: ("NREL_ZHAO_CHLORIDE_2020",) for p in ("density","specific_heat_capacity","dynamic_viscosity","thermal_conductivity")},
    "AIP_FLINAK_2017_V10": {
        "density": ("AIP_AN_CHENG_SU_ZHANG_2017",),
        "specific_heat_capacity": ("AIP_AN_CHENG_SU_ZHANG_2017",),
        "dynamic_viscosity": ("TASIDOU_ETAL_REFERENCE_VISCOSITY_2019",),
        "thermal_conductivity": ("AIP_AN_CHENG_SU_ZHANG_2017",),
    },
    "AIP_LINAK_CARBONATE_2017_V10": {
        "density": ("AIP_AN_CHENG_SU_ZHANG_2017",),
        "specific_heat_capacity": ("AIP_AN_CHENG_SU_ZHANG_2017",),
        "dynamic_viscosity": ("TASIDOU_ETAL_REFERENCE_VISCOSITY_2019",),
        "thermal_conductivity": ("AIP_AN_CHENG_SU_ZHANG_2017",),
    },
    "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10": {p: ("AIP_AN_CHENG_SU_ZHANG_2017",) for p in ("density","specific_heat_capacity","dynamic_viscosity","thermal_conductivity")},
}

_PROPERTY_UNCERTAINTY_95: dict[tuple[str, str], str] = {
    ("AIP_FLINAK_2017_V10", "dynamic_viscosity"): "2.9% (95% confidence; 2019 reference correlation)",
    ("AIP_LINAK_CARBONATE_2017_V10", "dynamic_viscosity"): "3.0% (95% confidence; 2019 reference correlation)",
    ("NREL_MGCL2_KCL_NACL_BASELINE_V10", "density"): "source-reported standard deviation within 1%",
    ("NREL_MGCL2_KCL_NACL_BASELINE_V10", "dynamic_viscosity"): "source-reported standard deviation 2-6%",
    ("NREL_MGCL2_KCL_NACL_BASELINE_V10", "thermal_conductivity"): "source-reported standard deviation 8-12%",
}

def property_source_ids(model_id: str, property_name: str) -> tuple[str, ...]:
    try:
        return _PROPERTY_SOURCE_MAP[model_id][property_name]
    except KeyError as exc:
        raise KeyError(f"No property provenance for {model_id}/{property_name}") from exc

def property_uncertainty_text(model_id: str, property_name: str) -> str:
    return _PROPERTY_UNCERTAINTY_95.get((model_id, property_name), "not fully quantified in registered source; retained as missing")


def model_specs() -> dict[str, FluidModelSpec]:
    return dict(_MODEL_SPECS)


def get_model_spec(model_id: str) -> FluidModelSpec:
    try:
        return _MODEL_SPECS[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown or evidence-blocked fluid model: {model_id}") from exc


def _assert_domain(spec: FluidModelSpec, temperature_C: float) -> None:
    if not (spec.valid_temperature_min_C <= temperature_C <= spec.valid_temperature_max_C):
        raise ValueError(
            f"{spec.model_id} evaluated outside registered domain: {temperature_C:.3f} C not in "
            f"[{spec.valid_temperature_min_C:.3f}, {spec.valid_temperature_max_C:.3f}] C"
        )


def fluid_properties(model_id: str, temperature_K: float) -> FluidProperties:
    spec = get_model_spec(model_id)
    T_C = float(temperature_K) - 273.15
    T_K = float(temperature_K)
    _assert_domain(spec, T_C)

    if model_id == "SOLAR_SALT_60_40_V10":
        rho = 2090.0 - 0.636 * T_C
        mu = 0.022714 - 0.120e-3 * T_C + 2.281e-7 * T_C**2 - 1.474e-10 * T_C**3
        k = 0.443 + 1.9e-4 * T_C
        cp = 1443.0 + 0.172 * T_C
    elif model_id == "NREL_MGCL2_KCL_NACL_BASELINE_V10":
        rho = -0.5878 * T_C + 1974.0
        mu = 0.689 * math.exp(1224.73 / T_K) * 1e-3  # cP -> Pa s
        k = 7.151e-7 * T_C**2 - 1.066e-3 * T_C + 0.811
        cp = (1.284e-6 * T_C**2 - 1.843e-3 * T_C + 1.661) * 1000.0  # J/g/K -> J/kg/K
    elif model_id == "AIP_FLINAK_2017_V10":
        rho = (2.45 - 6.53e-4 * T_C) * 1000.0
        mu = 0.2017 * math.exp(770.0 / T_K + 1_630_700.0 / T_K**2) * 1e-3  # reference mPa s -> Pa s
        k = -0.0635 + 1.40e-3 * T_C
        cp = 1.88 * 1000.0
    elif model_id == "AIP_LINAK_CARBONATE_2017_V10":
        rho = (2.27 - 4.34e-4 * T_C) * 1000.0
        mu = 0.1901 * math.exp(2660.0 / T_K + 726_990.0 / T_K**2) * 1e-3  # reference mPa s -> Pa s
        k = 0.336 + 2.58e-4 * T_C
        cp = 1.61 * 1000.0
    elif model_id == "AIP_LIF_NAK_CARBONATE_RECIPROCAL_2017_V10":
        rho = (2.31 - 4.13e-4 * T_C) * 1000.0
        mu = 0.0589 * math.exp(4132.9 / T_K) * 1e-3  # cP -> Pa s
        k = -0.135 + 1.58e-3 * T_C
        cp = 1.82 * 1000.0
    else:  # pragma: no cover
        raise AssertionError(model_id)

    vals = (rho, cp, mu, k)
    if not all(math.isfinite(v) and v > 0 for v in vals):
        raise ValueError(f"Non-physical property evaluation for {model_id} at {T_C:.3f} C: {vals}")
    return FluidProperties(
        model_id=model_id,
        temperature_C=T_C,
        density_kg_m3=rho,
        cp_J_kgK=cp,
        viscosity_Pa_s=mu,
        thermal_conductivity_W_mK=k,
        source_ids=spec.source_ids,
    )


def common_domain(model_ids: list[str]) -> tuple[float, float]:
    specs = [get_model_spec(m) for m in model_ids]
    lo = max(s.valid_temperature_min_C for s in specs)
    hi = min(s.valid_temperature_max_C for s in specs)
    if lo >= hi:
        raise ValueError(f"No non-zero common temperature domain for {model_ids}: [{lo}, {hi}] C")
    return lo, hi


def volumetric_sensible_energy_J_m3(model_id: str, cold_K: float, hot_K: float, n: int = 401) -> float:
    if hot_K <= cold_K:
        raise ValueError("hot_K must exceed cold_K")
    # Domain is checked at every quadrature node; this intentionally fails closed.
    T = np.linspace(float(cold_K), float(hot_K), int(n))
    rho_cp = np.empty_like(T)
    for i, t in enumerate(T):
        p = fluid_properties(model_id, float(t))
        rho_cp[i] = p.density_kg_m3 * p.cp_J_kgK
    return float(np.trapezoid(rho_cp, T))


def packed_bed_transport_v10(
    *, model_id: str, internal_diameter_m: float, bed_height_m: float, porosity: float,
    particle_diameter_m: float, mass_flow_kg_s: float, mean_temperature_K: float,
    solid_thermal_conductivity_W_mK: float,
    property_multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(solid_thermal_conductivity_W_mK) or solid_thermal_conductivity_W_mK <= 0:
        raise ValueError("solid_thermal_conductivity_W_mK must be finite and positive")
    props = fluid_properties(model_id, mean_temperature_K)
    multipliers = {"density": 1.0, "cp": 1.0, "viscosity": 1.0, "conductivity": 1.0}
    if property_multipliers:
        unknown = set(property_multipliers) - set(multipliers)
        if unknown:
            raise ValueError(f"Unknown property multipliers: {sorted(unknown)}")
        multipliers.update({k: float(v) for k, v in property_multipliers.items()})
    if not all(math.isfinite(v) and v > 0 for v in multipliers.values()):
        raise ValueError("Property multipliers must be finite and positive")
    rho = props.density_kg_m3 * multipliers["density"]
    cp = props.cp_J_kgK * multipliers["cp"]
    mu = props.viscosity_Pa_s * multipliers["viscosity"]
    conductivity = props.thermal_conductivity_W_mK * multipliers["conductivity"]
    area = math.pi * internal_diameter_m**2 / 4.0
    u = mass_flow_kg_s / (rho * area)
    re = rho * u * particle_diameter_m / mu
    pr = cp * mu / conductivity
    nu = gunn_nusselt_number(re, pr, porosity)
    h_conv = nu * conductivity / particle_diameter_m
    # The model includes the volume-mean spherical-particle conduction resistance
    # used by Trevisan et al. (J. Energy Storage 36, 102441, Eq. 5):
    # 1/h_eff = 1/h_conv + d_p/(10 k_s).  This is essential here because the
    # raw particle Biot number is not in the lumped-capacitance regime.
    particle_resistance = particle_diameter_m / (10.0 * solid_thermal_conductivity_W_mK)
    h_effective = 1.0 / (1.0 / h_conv + particle_resistance)
    particle_biot_raw = h_conv * (particle_diameter_m / 6.0) / solid_thermal_conductivity_W_mK
    a_s = 6.0 * (1.0 - porosity) / particle_diameter_m
    dp = ergun_pressure_drop_Pa(rho, mu, u, bed_height_m, porosity, particle_diameter_m)
    return {
        **props.as_dict(),
        "density_kg_m3": rho,
        "cp_J_kgK": cp,
        "viscosity_Pa_s": mu,
        "thermal_conductivity_W_mK": conductivity,
        "property_multipliers": multipliers,
        "cross_section_m2": area,
        "superficial_velocity_m_s": u,
        "particle_Re": re,
        "Pr": pr,
        "Nu_gunn": nu,
        "interphase_h_convective_W_m2K": h_conv,
        "particle_internal_resistance_m2K_W": particle_resistance,
        "interphase_h_effective_W_m2K": h_effective,
        "particle_Biot_raw": particle_biot_raw,
        "particle_resistance_correction_ratio": h_effective / h_conv,
        "specific_interphase_area_m2_m3": a_s,
        "volumetric_interphase_hA_W_m3K": h_effective * a_s,
        "pressure_drop_Pa": dp,
        "correlation_valid": bool(0.35 <= porosity < 1.0 and re <= 1.0e5 and pr > 0),
        "heat_transfer_correlation": "Gunn1978",
        "pressure_drop_correlation": "Ergun1952",
    }


def evaluate_packed_bed_v10_fast(
    *, model_id: str, design_id: str, internal_diameter_m: float, bed_height_m: float,
    porosity: float, particle_diameter_m: float, mass_flow_kg_s: float,
    cold_temperature_K: float, hot_temperature_K: float, solid_density_kg_m3: float,
    solid_cp_J_kgK: float, solid_thermal_conductivity_W_mK: float, n_cells: int = 24,
    property_multipliers: dict[str, float] | None = None,
) -> dict[str, Any]:
    if hot_temperature_K <= cold_temperature_K:
        raise ValueError("hot_temperature_K must exceed cold_temperature_K")
    # Checking both endpoints prevents a mean-temperature-only domain loophole.
    fluid_properties(model_id, cold_temperature_K)
    fluid_properties(model_id, hot_temperature_K)
    mean_T = 0.5 * (cold_temperature_K + hot_temperature_K)
    tr = packed_bed_transport_v10(
        model_id=model_id, internal_diameter_m=internal_diameter_m, bed_height_m=bed_height_m,
        porosity=porosity, particle_diameter_m=particle_diameter_m, mass_flow_kg_s=mass_flow_kg_s,
        mean_temperature_K=mean_T,
        solid_thermal_conductivity_W_mK=solid_thermal_conductivity_W_mK,
        property_multipliers=property_multipliers,
    )
    rho_f = float(tr["density_kg_m3"]); cp_f = float(tr["cp_J_kgK"])
    area = float(tr["cross_section_m2"]); H = float(bed_height_m); eps = float(porosity)
    V = area * H; delta = float(hot_temperature_K - cold_temperature_K)
    multipliers = tr["property_multipliers"]
    fluid_energy_density = volumetric_sensible_energy_J_m3(model_id, cold_temperature_K, hot_temperature_K)
    fluid_energy_density *= float(multipliers["density"]) * float(multipliers["cp"])
    # Use the integrated rho*Cp over the full registered duty as the dynamic volumetric capacitance.
    rho_cp_eff = fluid_energy_density / delta
    cap_f_vol = eps * rho_cp_eff
    cap_s_vol = (1.0 - eps) * solid_density_kg_m3 * solid_cp_J_kgK
    hA = float(tr["volumetric_interphase_hA_W_m3K"])
    u = float(tr["superficial_velocity_m_s"])
    dz = H / n_cells; cell_vol = area * dz; adv_velocity = u / eps
    tau_f = cap_f_vol / max(hA, 1e-12); tau_s = cap_s_vol / max(hA, 1e-12)
    stable = min(2.0, 0.35 * dz / max(adv_velocity, 1e-12), 0.20 * tau_f, 0.20 * tau_s)
    stable = max(stable, 1e-3)
    cold = float(cold_temperature_K); hot = float(hot_temperature_K)
    Q_span = (cap_f_vol + cap_s_vol) * V * delta
    nominal = Q_span / max(mass_flow_kg_s * cp_f * delta, 1e-12)
    # Internal particle resistance broadens the charge/discharge front; the
    # longer horizon avoids classifying physically slow cases as numerical failures.
    t_end = 2.0 * nominal
    n_steps = max(2, int(math.ceil(t_end / stable))); dt = t_end / n_steps
    ch = _fast_core(1, n_cells, n_steps, dt, dz, adv_velocity, cap_f_vol, cap_s_vol, hA, cell_vol, Q_span, hot, cold, cold, hot, mass_flow_kg_s, cp_f)
    dis = _fast_core(0, n_cells, n_steps, dt, dz, adv_velocity, cap_f_vol, cap_s_vol, hA, cell_vol, Q_span, cold, hot, cold, hot, mass_flow_kg_s, cp_f)
    capacity = Q_span / J_PER_MWH
    t90 = None if ch[2] < 0 else float(ch[2])
    pump_W = float(tr["pressure_drop_Pa"]) * (mass_flow_kg_s / rho_f)
    pump_fraction = None if t90 is None else pump_W * t90 / max(capacity * J_PER_MWH, 1e-12)
    charge_balance_error = 100.0 * (float(ch[7]) / max(float(ch[5]), 1e-12) - 1.0)
    discharge_balance_error = 100.0 * ((float(dis[5]) + float(dis[7])) / max(Q_span, 1e-12) - 1.0)
    spec = get_model_spec(model_id)
    return {
        "design_id": design_id,
        "fluid_model_id": model_id,
        "fluid_name": spec.material_name,
        "family": spec.family,
        "architecture": "sensible_packed_bed_LTNE_v10_particle_resistance",
        "evidence_status": "dynamic_eligible_only_within_registered_property_domain",
        "cold_temperature_C": cold - 273.15,
        "hot_temperature_C": hot - 273.15,
        "mass_flow_kg_s": mass_flow_kg_s,
        "internal_diameter_m": float(internal_diameter_m),
        "bed_height_m": float(bed_height_m),
        "porosity": float(porosity),
        "particle_diameter_m": float(particle_diameter_m),
        "n_cells": int(n_cells),
        "storage_volume_m3": V,
        "fluid_volumetric_sensible_kWh_m3": fluid_energy_density / 3.6e6,
        "stored_heat_MWh": capacity,
        "volumetric_capacity_MWh_m3": capacity / V,
        "charge_t50_h": None if ch[1] < 0 else float(ch[1]) / 3600.0,
        "charge_t90_h": None if t90 is None else t90 / 3600.0,
        "discharge_t50_h": None if dis[1] < 0 else float(dis[1]) / 3600.0,
        "discharge_t90_h": None if dis[2] < 0 else float(dis[2]) / 3600.0,
        "high_grade_discharge_duration_h": float(dis[8]) / 3600.0,
        "thermal_front_thickness_ratio": None if ch[3] < 0 else float(ch[3]) / H,
        "high_grade_discharge_fraction": float(dis[4] / max(Q_span, 1e-12)),
        "charge_energy_balance_error_pct": charge_balance_error,
        "discharge_energy_balance_error_pct": discharge_balance_error,
        "discharge_energy_recovery_fraction": float(dis[5] / max(Q_span, 1e-12)),
        "pressure_drop_kPa": float(tr["pressure_drop_Pa"]) / 1000.0,
        "pump_power_W": pump_W,
        "pump_energy_fraction": pump_fraction,
        "particle_Re": float(tr["particle_Re"]),
        "Pr": float(tr["Pr"]),
        "Nu": float(tr["Nu_gunn"]),
        "particle_Biot_raw": float(tr["particle_Biot_raw"]),
        "particle_resistance_correction_ratio": float(tr["particle_resistance_correction_ratio"]),
        "interphase_h_convective_W_m2K": float(tr["interphase_h_convective_W_m2K"]),
        "interphase_h_effective_W_m2K": float(tr["interphase_h_effective_W_m2K"]),
        "interphase_NTU": float(tr["volumetric_interphase_hA_W_m3K"]) * V / max(mass_flow_kg_s * cp_f, 1e-12),
        "particle_Peclet": float(tr["particle_Re"]) * float(tr["Pr"]),
        "mean_density_kg_m3": rho_f,
        "mean_cp_J_kgK": cp_f,
        "mean_viscosity_mPa_s": float(tr["viscosity_Pa_s"]) * 1000.0,
        "mean_k_W_mK": float(tr["thermal_conductivity_W_mK"]),
        "correlation_valid": bool(tr["correlation_valid"]),
        "source_ids": ";".join(spec.source_ids),
        "quantitative_use_class": spec.quantitative_use_class,
        "density_multiplier": float(multipliers["density"]),
        "cp_multiplier": float(multipliers["cp"]),
        "viscosity_multiplier": float(multipliers["viscosity"]),
        "conductivity_multiplier": float(multipliers["conductivity"]),
    }


def simulate_packed_bed_v10_detailed(
    *, model_id: str, design_id: str, mode: str, internal_diameter_m: float,
    bed_height_m: float, porosity: float, particle_diameter_m: float,
    mass_flow_kg_s: float, cold_temperature_K: float, hot_temperature_K: float,
    solid_density_kg_m3: float, solid_cp_J_kgK: float,
    solid_thermal_conductivity_W_mK: float, n_cells: int = 64,
    max_snapshots: int = 81,
) -> dict[str, Any]:
    """Return inspectable temperature fields for selected frozen-property designs.

    This routine mirrors the production Schumann/LTNE kernel, including the
    spherical-particle conduction resistance.  It is intentionally reserved for
    a few representative designs so the full optimization remains fast while the
    plotted transient fields stay traceable to the governing equations.
    """
    if mode not in {"charge", "discharge"}:
        raise ValueError("mode must be 'charge' or 'discharge'")
    if n_cells < 4 or max_snapshots < 3:
        raise ValueError("detailed simulations require at least four cells and three snapshots")
    fluid_properties(model_id, cold_temperature_K)
    fluid_properties(model_id, hot_temperature_K)
    mean_T = 0.5 * (cold_temperature_K + hot_temperature_K)
    tr = packed_bed_transport_v10(
        model_id=model_id,
        internal_diameter_m=internal_diameter_m,
        bed_height_m=bed_height_m,
        porosity=porosity,
        particle_diameter_m=particle_diameter_m,
        mass_flow_kg_s=mass_flow_kg_s,
        mean_temperature_K=mean_T,
        solid_thermal_conductivity_W_mK=solid_thermal_conductivity_W_mK,
    )
    cold = float(cold_temperature_K)
    hot = float(hot_temperature_K)
    delta = hot - cold
    if delta <= 0:
        raise ValueError("hot_temperature_K must exceed cold_temperature_K")
    eps = float(porosity)
    area = float(tr["cross_section_m2"])
    height = float(bed_height_m)
    volume = area * height
    dz = height / n_cells
    z = (np.arange(n_cells, dtype=float) + 0.5) * dz
    cell_volume = area * dz
    fluid_energy_density = volumetric_sensible_energy_J_m3(model_id, cold, hot)
    rho_cp_eff = fluid_energy_density / delta
    cap_f_vol = eps * rho_cp_eff
    cap_s_vol = (1.0 - eps) * solid_density_kg_m3 * solid_cp_J_kgK
    hA = float(tr["volumetric_interphase_hA_W_m3K"])
    adv_velocity = float(tr["superficial_velocity_m_s"]) / eps
    tau_f = cap_f_vol / max(hA, 1e-12)
    tau_s = cap_s_vol / max(hA, 1e-12)
    stable = min(2.0, 0.35 * dz / max(adv_velocity, 1e-12), 0.20 * tau_f, 0.20 * tau_s)
    stable = max(stable, 1e-3)
    cp_f = float(tr["cp_J_kgK"])
    q_span = (cap_f_vol + cap_s_vol) * volume * delta
    nominal = q_span / max(mass_flow_kg_s * cp_f * delta, 1e-12)
    t_end = 2.0 * nominal
    n_steps = max(2, int(math.ceil(t_end / stable)))
    dt = t_end / n_steps
    record_steps = set(np.linspace(0, n_steps, max_snapshots, dtype=int).tolist())

    initial = cold if mode == "charge" else hot
    inlet = hot if mode == "charge" else cold
    t_f = np.full(n_cells, initial, dtype=float)
    t_s = np.full(n_cells, initial, dtype=float)
    time_s = 0.0
    boundary_energy_j = 0.0
    high_grade_energy_j = 0.0
    high_grade_duration_s = 0.0
    high_grade_temperature = cold + 0.9 * delta
    fields: list[dict[str, float | str | int]] = []
    trace: list[dict[str, float | str | int]] = []

    def inventory_fraction() -> float:
        inventory = (
            cap_f_vol * np.sum(t_f - cold) * cell_volume
            + cap_s_vol * np.sum(t_s - cold) * cell_volume
        )
        return float(np.clip(inventory / max(q_span, 1e-12), 0.0, 1.0))

    def record(snapshot: int) -> None:
        inv = inventory_fraction()
        trace.append({
            "design_id": design_id,
            "fluid_model_id": model_id,
            "mode": mode,
            "snapshot": snapshot,
            "time_h": time_s / 3600.0,
            "dimensionless_time": time_s / max(nominal, 1e-12),
            "outlet_theta": float((t_f[-1] - cold) / delta),
            "inventory_fraction": inv,
            "extracted_fraction": 1.0 - inv,
            "max_fluid_solid_lag_K": float(np.max(np.abs(t_f - t_s))),
        })
        for cell in range(n_cells):
            fields.append({
                "design_id": design_id,
                "fluid_model_id": model_id,
                "mode": mode,
                "snapshot": snapshot,
                "time_h": time_s / 3600.0,
                "dimensionless_time": time_s / max(nominal, 1e-12),
                "cell": cell,
                "z_over_H": float(z[cell] / height),
                "fluid_theta": float((t_f[cell] - cold) / delta),
                "solid_theta": float((t_s[cell] - cold) / delta),
            })

    snapshot = 0
    record(snapshot)
    for step in range(1, n_steps + 1):
        upstream = np.empty_like(t_f)
        upstream[0] = inlet
        upstream[1:] = t_f[:-1]
        adv = -adv_velocity * (t_f - upstream) / dz
        q_fs = hA * (t_s - t_f)
        t_f = np.clip(t_f + dt * (adv + q_fs / cap_f_vol), cold, hot)
        t_s = np.clip(t_s - dt * q_fs / cap_s_vol, cold, hot)
        time_s += dt
        outlet = float(t_f[-1])
        if mode == "charge":
            boundary_energy_j += mass_flow_kg_s * cp_f * max(inlet - outlet, 0.0) * dt
        else:
            power = mass_flow_kg_s * cp_f * max(outlet - cold, 0.0)
            boundary_energy_j += power * dt
            if outlet >= high_grade_temperature:
                high_grade_energy_j += power * dt
                high_grade_duration_s = time_s
        if step in record_steps:
            snapshot += 1
            record(snapshot)

    inv_final = inventory_fraction()
    if mode == "charge":
        balance_error = 100.0 * (inv_final * q_span / max(boundary_energy_j, 1e-12) - 1.0)
    else:
        balance_error = 100.0 * ((boundary_energy_j + inv_final * q_span) / max(q_span, 1e-12) - 1.0)
    return {
        "summary": {
            "design_id": design_id,
            "fluid_model_id": model_id,
            "mode": mode,
            "n_cells": n_cells,
            "n_steps": n_steps,
            "time_step_s": dt,
            "nominal_time_h": nominal / 3600.0,
            "simulated_time_h": time_s / 3600.0,
            "energy_balance_error_pct": balance_error,
            "high_grade_discharge_duration_h": high_grade_duration_s / 3600.0,
            "high_grade_discharge_fraction": high_grade_energy_j / max(q_span, 1e-12),
            "particle_Biot_raw": float(tr["particle_Biot_raw"]),
            "particle_resistance_correction_ratio": float(tr["particle_resistance_correction_ratio"]),
        },
        "trace": trace,
        "fields": fields,
    }
