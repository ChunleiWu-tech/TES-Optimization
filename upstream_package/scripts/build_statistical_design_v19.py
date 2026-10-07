"""Build the statistical-design extension from frozen V18 results.

The analysis deliberately separates two questions:

1. Cluster bootstrap intervals describe sensitivity to the sampled base-geometry
   design space.  They are not material-property or experimental uncertainty.
2. Integer-boundary stresses are deterministic worst-case numerical
   perturbations.  They are not probabilistic intervals.

No transient simulation is rerun and no physical-model parameter is changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_V18 = ROOT / "results_v18"
RESULTS_V19 = ROOT / "results_v19"
CONFIG_PATH = ROOT / "config" / "study_v18.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ceil_array(values: np.ndarray, relative_tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    nearest = np.rint(values)
    scale = np.maximum(1.0, np.abs(values))
    adjusted = np.where(
        np.abs(values - nearest) <= relative_tolerance * scale,
        nearest,
        values,
    )
    return np.ceil(adjusted).astype(np.int64)


def _classification_changes_possible(
    n_i: np.ndarray, n_j: np.ndarray, epsilon: float, relative_tolerance: float,
) -> np.ndarray:
    """Check all attainable integer counts over the full perturbation box.

    Ceiling is monotone, so continuous intervals map to integer intervals.
    An unequal pair can become equal if these integer intervals overlap.
    An equal pair stays equal only if both contain the same single count.
    Four corners alone do not suffice because equality is not monotone.
    """
    if epsilon < 0:
        raise ValueError("Perturbation magnitude must be nonnegative")
    baseline_equal = _ceil_array(n_i, relative_tolerance) == _ceil_array(n_j, relative_tolerance)
    lower_i = _ceil_array(n_i - epsilon, relative_tolerance)
    upper_i = _ceil_array(n_i + epsilon, relative_tolerance)
    lower_j = _ceil_array(n_j - epsilon, relative_tolerance)
    upper_j = _ceil_array(n_j + epsilon, relative_tolerance)
    equality_possible = np.maximum(lower_i, lower_j) <= np.minimum(upper_i, upper_j)
    equality_unavoidable = (
        (lower_i == upper_i) & (lower_j == upper_j) & (lower_i == lower_j)
    )
    return np.where(baseline_equal, ~equality_unavoidable, equality_possible)


def _cluster_arrays(frame: pd.DataFrame, column: str, cluster_ids: np.ndarray) -> np.ndarray:
    grouped = frame.groupby("base_design_id", sort=True)[column]
    arrays = [grouped.get_group(cluster).to_numpy() for cluster in cluster_ids]
    sizes = {len(x) for x in arrays}
    if len(sizes) != 1:
        raise ValueError(f"Unequal within-cluster row counts for {column}: {sorted(sizes)}")
    return np.stack(arrays)


def _cluster_means(frame: pd.DataFrame, column: str, cluster_ids: np.ndarray) -> np.ndarray:
    return (
        frame.groupby("base_design_id", sort=True)[column]
        .mean()
        .reindex(cluster_ids)
        .to_numpy(float)
    )


def _bootstrap_median(
    matrix: np.ndarray,
    sample_indices: np.ndarray,
    batch_size: int = 100,
) -> np.ndarray:
    out = np.empty(len(sample_indices), dtype=float)
    for start in range(0, len(sample_indices), batch_size):
        stop = min(start + batch_size, len(sample_indices))
        sampled = matrix[sample_indices[start:stop]].reshape(stop - start, -1)
        out[start:stop] = np.nanmedian(sampled, axis=1)
    return out


def _interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    return tuple(np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0]).astype(float))


def _estimand_row(
    *,
    estimand: str,
    point: float,
    replicates: np.ndarray,
    unit: str,
    n_observations: int,
    n_clusters: int,
    confidence_level: float,
    scope: str,
) -> dict[str, object]:
    low, high = _interval(replicates, confidence_level)
    return {
        "estimand": estimand,
        "point_estimate": float(point),
        "confidence_low": low,
        "confidence_high": high,
        "unit": unit,
        "confidence_level": confidence_level,
        "bootstrap_method": "percentile cluster bootstrap",
        "independent_resampling_unit": "base_design_id",
        "n_observations": int(n_observations),
        "n_clusters": int(n_clusters),
        "scope": scope,
    }


def _attenuation_decomposition(pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stages = [
        ("fluid_to_filled_module", "A_fluid_log", "A_filled_log"),
        ("filled_module_to_transient_module", "A_filled_log", "A_dynamic_log"),
        ("transient_module_to_modular_system", "A_dynamic_log", "A_system_log"),
    ]
    records: list[pd.DataFrame] = []
    for order, (stage, source, target) in enumerate(stages, start=1):
        block = pairwise[
            [
                "group_id",
                "pair_id",
                "scenario_id",
                "base_design_id",
                "topology_id",
                source,
                target,
            ]
        ].copy()
        block = block.rename(columns={source: "signed_log_contrast_in", target: "signed_log_contrast_out"})
        block["stage_order"] = order
        block["stage"] = stage
        block["absolute_log_contrast_in"] = block["signed_log_contrast_in"].abs()
        block["absolute_log_contrast_out"] = block["signed_log_contrast_out"].abs()
        block["absolute_log_contrast_change"] = (
            block["absolute_log_contrast_out"] - block["absolute_log_contrast_in"]
        )
        block["attenuation_log_magnitude"] = -block["absolute_log_contrast_change"]
        block["stage_fate"] = np.select(
            [
                block["absolute_log_contrast_change"] < -1e-14,
                block["absolute_log_contrast_change"] > 1e-14,
            ],
            ["attenuation", "amplification"],
            default="unchanged",
        )
        records.append(block)
    long = pd.concat(records, ignore_index=True)
    summary = (
        long.groupby(["stage_order", "stage"], sort=True)
        .agg(
            comparisons=("pair_id", "size"),
            median_absolute_log_contrast_in=("absolute_log_contrast_in", "median"),
            median_absolute_log_contrast_out=("absolute_log_contrast_out", "median"),
            median_attenuation_log_magnitude=("attenuation_log_magnitude", "median"),
            attenuation_fraction=("stage_fate", lambda x: float(np.mean(x == "attenuation"))),
            amplification_fraction=("stage_fate", lambda x: float(np.mean(x == "amplification"))),
            unchanged_fraction=("stage_fate", lambda x: float(np.mean(x == "unchanged"))),
        )
        .reset_index()
    )
    return long, summary


def _pareto_release_stability(results: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [
        ("exact_nondominated_set", "system_pareto_exact_audit_v16.csv", "system_pareto_exact_audit_v18.csv"),
        ("objective_resolution_set", "system_pareto_front_v16.csv", "system_pareto_front_v18.csv"),
    ]
    for definition, old_name, new_name in pairs:
        old = pd.read_csv(results / old_name)
        new = pd.read_csv(results / new_name)
        scenarios = sorted(set(old["scenario_id"]) | set(new["scenario_id"]))
        for scenario in scenarios + ["ALL"]:
            old_part = old if scenario == "ALL" else old[old["scenario_id"] == scenario]
            new_part = new if scenario == "ALL" else new[new["scenario_id"] == scenario]
            key_columns = ["scenario_id", "design_id"]
            old_keys = set(map(tuple, old_part[key_columns].astype(str).to_numpy()))
            new_keys = set(map(tuple, new_part[key_columns].astype(str).to_numpy()))
            intersection = len(old_keys & new_keys)
            union = len(old_keys | new_keys)
            rows.append(
                {
                    "front_definition": definition,
                    "scenario_id": scenario,
                    "v16_count": len(old_keys),
                    "v18_count": len(new_keys),
                    "shared_count": intersection,
                    "union_count": union,
                    "jaccard_similarity": intersection / union if union else 1.0,
                    "v18_retained_from_v16_fraction": intersection / len(old_keys) if old_keys else 1.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    design = config["statistical_design"]
    replicates = int(design["bootstrap_replicates"])
    seed = int(design["bootstrap_seed"])
    confidence_level = float(design["confidence_level"])
    expected_clusters = int(design["cluster_count"])
    ceiling_tolerance = float(
        config["design_consequence_boundary"]["integer_ceiling_relative_tolerance"]
    )

    advantage = pd.read_csv(RESULTS_V18 / "advantage_transfer_v18.csv")
    pairwise = pd.read_csv(RESULTS_V18 / "02_pairwise_design_consequence.csv")
    integer_audit = pd.read_csv(RESULTS_V18 / "00_integer_reclosure_audit.csv")
    continuous = pd.read_csv(
        RESULTS_V18 / "01_continuous_service_closure.csv",
        usecols=["case_id", "base_design_id"],
    )
    integer_audit = integer_audit.merge(
        continuous.drop_duplicates("case_id"),
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if integer_audit["base_design_id"].isna().any():
        raise ValueError("Integer-reclosure rows could not be assigned to base-design clusters")

    cluster_ids = np.array(sorted(pairwise["base_design_id"].unique()))
    if len(cluster_ids) != expected_clusters:
        raise ValueError(f"Expected {expected_clusters} base geometries, found {len(cluster_ids)}")
    for frame_name, frame in (
        ("advantage", advantage),
        ("pairwise", pairwise),
        ("integer_audit", integer_audit),
    ):
        missing = set(cluster_ids) - set(frame["base_design_id"].unique())
        if missing:
            raise ValueError(f"{frame_name} is missing base geometries: {sorted(missing)[:5]}")

    RESULTS_V19.mkdir(parents=True, exist_ok=True)

    attenuation_long, attenuation_summary = _attenuation_decomposition(pairwise)
    attenuation_long.to_csv(RESULTS_V19 / "scale_attenuation_decomposition_v19.csv", index=False)
    attenuation_summary.to_csv(RESULTS_V19 / "scale_attenuation_summary_v19.csv", index=False)

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(cluster_ids), size=(replicates, len(cluster_ids)))

    solar_matrix = _cluster_arrays(advantage, "ARI_material_to_system", cluster_ids)
    pair_matrix = _cluster_arrays(pairwise, "R_system_fluid", cluster_ids)
    solar_boot = 100.0 * _bootstrap_median(solar_matrix, sample_indices)
    pair_boot = 100.0 * _bootstrap_median(pair_matrix, sample_indices)

    tie_all_cluster = _cluster_means(pairwise.assign(same_integer=pairwise["same_integer"].astype(float)), "same_integer", cluster_ids)
    tie_s10_frame = pairwise[pairwise["scenario_id"] == "S10MW_8H"].copy()
    tie_s50_frame = pairwise[pairwise["scenario_id"] == "S50MW_6H"].copy()
    tie_s10_cluster = _cluster_means(tie_s10_frame.assign(same_integer=tie_s10_frame["same_integer"].astype(float)), "same_integer", cluster_ids)
    tie_s50_cluster = _cluster_means(tie_s50_frame.assign(same_integer=tie_s50_frame["same_integer"].astype(float)), "same_integer", cluster_ids)
    correction_frame = integer_audit.assign(
        count_changed=(integer_audit["count_delta_vs_frozen"].astype(int) != 0).astype(float)
    )
    correction_cluster = _cluster_means(correction_frame, "count_changed", cluster_ids)

    def resampled_cluster_mean(values: np.ndarray) -> np.ndarray:
        return values[sample_indices].mean(axis=1)

    tie_all_boot = 100.0 * resampled_cluster_mean(tie_all_cluster)
    tie_s10_boot = 100.0 * resampled_cluster_mean(tie_s10_cluster)
    tie_s50_boot = 100.0 * resampled_cluster_mean(tie_s50_cluster)
    tie_difference_boot = tie_s50_boot - tie_s10_boot
    correction_boot = 100.0 * resampled_cluster_mean(correction_cluster)

    bootstrap = pd.DataFrame(
        {
            "replicate": np.arange(1, replicates + 1),
            "solar_reference_median_retention_pct": solar_boot,
            "all_pair_median_retention_pct": pair_boot,
            "same_integer_frequency_all_pct": tie_all_boot,
            "same_integer_frequency_S10MW_8H_pct": tie_s10_boot,
            "same_integer_frequency_S50MW_6H_pct": tie_s50_boot,
            "same_integer_frequency_difference_S50_minus_S10_pct": tie_difference_boot,
            "integer_reclosure_changed_frequency_pct": correction_boot,
        }
    )
    bootstrap.to_csv(RESULTS_V19 / "cluster_bootstrap_replicates_v19.csv", index=False)

    scope = (
        "Sensitivity to the sampled base-geometry design space; material pairs, topologies, "
        "and services remain paired within resampled clusters."
    )
    estimands = [
        _estimand_row(
            estimand="Median retained Solar-Salt-referenced contrast",
            point=100.0 * float(advantage["ARI_material_to_system"].median()),
            replicates=solar_boot,
            unit="%",
            n_observations=len(advantage),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
        _estimand_row(
            estimand="Median retained contrast across all material pairs",
            point=100.0 * float(pairwise["R_system_fluid"].median()),
            replicates=pair_boot,
            unit="%",
            n_observations=len(pairwise),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
        _estimand_row(
            estimand="Equal minimum-module-count frequency, all services",
            point=100.0 * float(pairwise["same_integer"].mean()),
            replicates=tie_all_boot,
            unit="%",
            n_observations=len(pairwise),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
        _estimand_row(
            estimand="Equal minimum-module-count frequency, 10 MW and 80 MWh",
            point=100.0 * float(tie_s10_frame["same_integer"].mean()),
            replicates=tie_s10_boot,
            unit="%",
            n_observations=len(tie_s10_frame),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
        _estimand_row(
            estimand="Equal minimum-module-count frequency, 50 MW and 300 MWh",
            point=100.0 * float(tie_s50_frame["same_integer"].mean()),
            replicates=tie_s50_boot,
            unit="%",
            n_observations=len(tie_s50_frame),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
        _estimand_row(
            estimand="Paired service difference in equal-count frequency, 50 MW minus 10 MW",
            point=100.0 * float(tie_s50_frame["same_integer"].mean() - tie_s10_frame["same_integer"].mean()),
            replicates=tie_difference_boot,
            unit="percentage points",
            n_observations=len(pairwise),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
        _estimand_row(
            estimand="Integer-reclosure count-change frequency",
            point=100.0 * float(correction_frame["count_changed"].mean()),
            replicates=correction_boot,
            unit="%",
            n_observations=len(integer_audit),
            n_clusters=len(cluster_ids),
            confidence_level=confidence_level,
            scope=scope,
        ),
    ]
    estimand_table = pd.DataFrame(estimands)
    estimand_table.to_csv(RESULTS_V19 / "statistical_estimands_v19.csv", index=False)

    n_i = pairwise["N_cont_i"].to_numpy(float)
    n_j = pairwise["N_cont_j"].to_numpy(float)
    perturbation_rows: list[dict[str, object]] = []
    for epsilon in map(float, design["integer_boundary_perturbations_module_count"]):
        unstable = _classification_changes_possible(n_i, n_j, epsilon, ceiling_tolerance)
        for scenario in list(sorted(pairwise["scenario_id"].unique())) + ["ALL"]:
            mask = np.ones(len(pairwise), dtype=bool) if scenario == "ALL" else pairwise["scenario_id"].to_numpy() == scenario
            perturbation_rows.append(
                {
                    "perturbation_module_count": epsilon,
                    "scenario_id": scenario,
                    "comparisons": int(mask.sum()),
                    "classification_changes_possible": int(unstable[mask].sum()),
                    "classification_stable_fraction": float(1.0 - unstable[mask].mean()),
                    "stress_definition": "full independent perturbation intervals of both continuous module counts; exact attainable integer-range overlap test",
                }
            )
    stability = pd.DataFrame(perturbation_rows)
    stability.to_csv(RESULTS_V19 / "integer_boundary_stability_v19.csv", index=False)

    pareto = _pareto_release_stability(RESULTS_V18)
    pareto.to_csv(RESULTS_V19 / "pareto_release_stability_v19.csv", index=False)

    summary = {
        "analysis_release": design["analysis_release"],
        "status": "PASS",
        "source_release": "V18 frozen deterministic results",
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "clusters": len(cluster_ids),
            "independent_resampling_unit": design["independent_resampling_unit"],
            "interpretation": design["interpretation"],
        },
        "quality_checks": {
            "all_boundary_identities_match": bool(pairwise["boundary_identity_match"].all()),
            "all_baseline_ties_match": bool(pairwise["baseline_tie_match"].all()),
            "no_pairwise_rank_reversals": bool((pairwise["A_fluid_log"] * pairwise["A_system_log"] >= 0).all()),
            "attenuation_stage_rows": int(len(attenuation_long)),
            "bootstrap_rows": int(len(bootstrap)),
            "estimand_rows": int(len(estimand_table)),
        },
        "key_results": {
            "solar_reference_retention_pct": estimands[0],
            "all_pair_retention_pct": estimands[1],
            "equal_count_frequency_pct": estimands[2],
            "paired_service_difference_percentage_points": estimands[5],
            "minimum_tested_perturbation_with_any_possible_change": (
                float(stability.loc[(stability["scenario_id"] == "ALL") & (stability["classification_changes_possible"] > 0), "perturbation_module_count"].min())
                if ((stability["scenario_id"] == "ALL") & (stability["classification_changes_possible"] > 0)).any()
                else None
            ),
            "objective_resolution_front_jaccard_v16_v18": float(
                pareto.loc[
                    (pareto["front_definition"] == "objective_resolution_set")
                    & (pareto["scenario_id"] == "ALL"),
                    "jaccard_similarity",
                ].iloc[0]
            ),
        },
    }
    summary_path = RESULTS_V19 / "statistical_design_summary_v19.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    excluded_from_manifest = {
        "statistical_design_manifest_v19.json",
        "unit_test_report_v19.json",
    }
    output_files = sorted(
        path
        for path in RESULTS_V19.iterdir()
        if path.is_file() and path.name not in excluded_from_manifest
    )
    manifest = {
        "analysis_release": design["analysis_release"],
        "source_directory": str(RESULTS_V18),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in output_files
        ],
    }
    (RESULTS_V19 / "statistical_design_manifest_v19.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "results": str(RESULTS_V19), "files": len(output_files) + 1}, indent=2))


if __name__ == "__main__":
    main()
