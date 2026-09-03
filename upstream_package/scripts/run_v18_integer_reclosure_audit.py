"""Re-evaluate the frozen V16 integer boundary with the archived V16 kernel.

The audit is intentionally independent of the continuous-coordinate solver.
For every service/design case it evaluates the frozen reported count and the
nearby integer counts with the current locked source snapshot.  This detects
whether a frozen count is still the minimum feasible integer before any V18
result is promoted.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tespub.codesign_v16 import _registered_ambient_temperature_K, load_config  # noqa: E402
from tespub.fluid_properties import fluid_properties  # noqa: E402
from tespub.ltne_v16 import BedGeometryV16, evaluate_packed_bed_v16  # noqa: E402


BASELINE = ROOT / "results_v16"
OUTPUT = ROOT / "results_v18" / "00_integer_reclosure_audit.csv"
CHECKPOINT = ROOT / ".v18_checkpoints" / "integer_reclosure_audit.partial.csv"
CONFIG = ROOT / "config" / "study_v18.json"


def _case_id(service: dict[str, Any]) -> str:
    return f"{service['scenario_id']}::{service['design_id']}"


def _audit_case(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], int]
) -> dict[str, Any]:
    source, service, scenario, config, search_span = task
    cold_K, hot_K = [float(x) + 273.15 for x in config["common_duty_C"]]
    delta_t = hot_K - cold_K
    power_MW = float(scenario["target_thermal_power_MW"])
    required_energy = power_MW * float(scenario["storage_duration_h"])
    multipliers = {
        "density": float(source.get("density_multiplier", 1.0)),
        "cp": float(source.get("cp_multiplier", 1.0)),
        "viscosity": float(source.get("viscosity_multiplier", 1.0)),
        "conductivity": float(source.get("conductivity_multiplier", 1.0)),
    }
    transport_temperature = scenario.get("transport_property_temperature_K")
    if transport_temperature is None:
        cp = (
            float(source["mean_cp_J_kgK"])
            if "mean_cp_J_kgK" in source and pd.notna(source["mean_cp_J_kgK"])
            else fluid_properties(
                str(source["fluid_model_id"]), 0.5 * (cold_K + hot_K)
            ).cp_J_kgK
            * multipliers["cp"]
        )
    else:
        cp = (
            fluid_properties(
                str(source["fluid_model_id"]), float(transport_temperature)
            ).cp_J_kgK
            * multipliers["cp"]
        )
    total_flow = power_MW * 1e6 / (cp * delta_t)
    unit_power_MW = float(source["nominal_thermal_input_kW"]) / 1000.0
    n_power = int(math.ceil(power_MW / max(unit_power_MW, 1e-12)))
    n_frozen = int(service["parallel_unit_count"])
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
    cache: dict[int, dict[str, Any]] = {}

    def evaluate(count: int) -> dict[str, Any]:
        count = int(count)
        if count not in cache:
            actual = evaluate_packed_bed_v16(
                model_id=str(source["fluid_model_id"]),
                design_id=f"{source['design_id']}__{scenario['scenario_id']}__V18_INTEGER_AUDIT_N{count}",
                geometry=geometry,
                mass_flow_kg_s=total_flow / count,
                cold_temperature_K=cold_K,
                hot_temperature_K=hot_K,
                solid_density_kg_m3=float(solid["solid_density_kg_m3"]),
                solid_cp_J_kgK=float(solid["solid_cp_J_kgK"]),
                solid_thermal_conductivity_W_mK=float(
                    solid["solid_thermal_conductivity_W_mK"]
                ),
                n_cells=int(source["n_cells"]),
                axial_dispersion_m2_s=float(source.get("axial_dispersion_m2_s", 0.0)),
                solid_axial_diffusivity_m2_s=float(
                    source.get("solid_axial_diffusivity_m2_s", 0.0)
                ),
                wall_loss_W_m3K=float(source.get("wall_loss_W_m3K", 0.0)),
                ambient_temperature_K=_registered_ambient_temperature_K(source, config),
                minimum_outlet_grade_fraction=float(
                    scenario["minimum_outlet_grade_fraction"]
                ),
                property_multipliers=multipliers,
                transport_property_temperature_K=(
                    None
                    if transport_temperature is None
                    else float(transport_temperature)
                ),
            )
            installed = count * float(actual["stored_heat_MWh"]) * float(
                actual["high_grade_discharge_fraction"]
            )
            actual["audit_installed_energy_MWh"] = installed
            actual["audit_feasible"] = bool(installed + 1e-12 >= required_energy)
            cache[count] = actual
        return cache[count]

    lower_limit = max(n_power, n_frozen - search_span)
    upper_limit = n_frozen + search_span
    evaluated_counts = list(range(lower_limit, upper_limit + 1))
    for count in evaluated_counts:
        evaluate(count)
    feasible_counts = [count for count in evaluated_counts if cache[count]["audit_feasible"]]
    current_minimum = min(feasible_counts) if feasible_counts else np.nan
    lower_truncated = bool(
        feasible_counts and current_minimum == lower_limit and lower_limit > n_power
    )
    upper_truncated = bool(not feasible_counts)
    capacities = np.array(
        [float(cache[count]["audit_installed_energy_MWh"]) for count in evaluated_counts],
        dtype=float,
    )
    nondecreasing = bool(np.all(np.diff(capacities) >= -1e-10))
    frozen_actual = cache[n_frozen]
    adjacent_count = max(n_power, n_frozen - 1)
    adjacent_actual = cache[adjacent_count]
    corrected_actual = (
        cache[int(current_minimum)] if np.isfinite(current_minimum) else frozen_actual
    )
    return {
        "case_id": _case_id(service),
        "scenario_id": str(service["scenario_id"]),
        "design_id": str(service["design_id"]),
        "fluid_model_id": str(source["fluid_model_id"]),
        "frozen_parallel_unit_count": n_frozen,
        "power_lower_bound_count": n_power,
        "current_minimum_count": current_minimum,
        "count_delta_vs_frozen": (
            float(current_minimum - n_frozen) if np.isfinite(current_minimum) else np.nan
        ),
        "frozen_count_feasible": bool(frozen_actual["audit_feasible"]),
        "adjacent_lower_count": adjacent_count,
        "adjacent_lower_feasible": bool(adjacent_actual["audit_feasible"]),
        "frozen_installed_energy_MWh_recomputed": float(
            frozen_actual["audit_installed_energy_MWh"]
        ),
        "frozen_installed_energy_MWh_reported": float(
            service["installed_high_grade_capacity_MWh"]
        ),
        "frozen_capacity_reproduction_error_pct": 100.0
        * (
            float(frozen_actual["audit_installed_energy_MWh"])
            / float(service["installed_high_grade_capacity_MWh"])
            - 1.0
        ),
        "adjacent_lower_energy_margin_pct": 100.0
        * (
            float(adjacent_actual["audit_installed_energy_MWh"]) / required_energy
            - 1.0
        ),
        "corrected_installed_energy_MWh": float(
            corrected_actual["audit_installed_energy_MWh"]
        ),
        "corrected_capacity_margin_pct": 100.0
        * (
            float(corrected_actual["audit_installed_energy_MWh"]) / required_energy
            - 1.0
        ),
        "local_integer_capacity_nondecreasing": nondecreasing,
        "lower_search_truncated": lower_truncated,
        "upper_search_truncated": upper_truncated,
        "search_span_counts": search_span,
        "evaluations": len(cache),
        "audit_kernel": "archived V16 LTNE source with study_v18 numerical settings inherited unchanged from study_v16",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--search-span", type=int, default=2)
    parser.add_argument("--checkpoint-rows", type=int, default=32)
    args = parser.parse_args()
    unit = pd.read_csv(BASELINE / "codesign_results_v16.csv")
    service = pd.read_csv(BASELINE / "fixed_service_results_v16.csv")
    config = load_config(CONFIG)
    scenarios = {str(row["scenario_id"]): row for row in config["service_scenarios"]}
    unit_index = unit.set_index("design_id", drop=False)
    tasks = [
        (
            unit_index.loc[str(row["design_id"])].to_dict(),
            row,
            scenarios[str(row["scenario_id"])],
            config,
            int(args.search_span),
        )
        for row in service.to_dict("records")
    ]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(CHECKPOINT) if CHECKPOINT.is_file() else pd.DataFrame()
    completed = set(previous["case_id"].astype(str)) if not previous.empty else set()
    pending = [task for task in tasks if _case_id(task[1]) not in completed]
    start = time.perf_counter()
    processed = len(completed)
    for offset in range(0, len(pending), int(args.checkpoint_rows)):
        batch = pending[offset : offset + int(args.checkpoint_rows)]
        if int(args.workers) > 1 and len(batch) > 1:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                rows = list(executor.map(_audit_case, batch, chunksize=1))
        else:
            rows = [_audit_case(task) for task in batch]
        frame = pd.DataFrame(rows)
        frame.to_csv(
            CHECKPOINT,
            mode="a",
            header=not CHECKPOINT.exists() or CHECKPOINT.stat().st_size == 0,
            index=False,
        )
        processed += len(frame)
        elapsed = time.perf_counter() - start
        rate = (processed - len(completed)) / max(elapsed, 1e-9)
        eta_h = (len(tasks) - processed) / max(rate, 1e-9) / 3600.0
        print(
            f"[V18 integer reclosure] {processed}/{len(tasks)} persisted; "
            f"elapsed={elapsed/3600.0:.2f} h; eta={eta_h:.2f} h",
            flush=True,
        )
    result = pd.read_csv(CHECKPOINT)
    result = result.drop_duplicates("case_id", keep="last").sort_values(
        ["scenario_id", "design_id"]
    )
    result.to_csv(OUTPUT, index=False)
    failures = result[
        (~result["frozen_count_feasible"].astype(bool))
        | result["lower_search_truncated"].astype(bool)
        | result["upper_search_truncated"].astype(bool)
        | (~result["local_integer_capacity_nondecreasing"].astype(bool))
    ]
    print(
        {
            "rows": int(len(result)),
            "count_changes": int((result["count_delta_vs_frozen"] != 0).sum()),
            "frozen_infeasible": int((~result["frozen_count_feasible"].astype(bool)).sum()),
            "audit_failures": int(len(failures)),
            "maximum_absolute_capacity_reproduction_error_pct": float(
                result["frozen_capacity_reproduction_error_pct"].abs().max()
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
