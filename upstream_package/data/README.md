# V16 data directory

- `authoritative_source_registry.csv`: source identity, authority tier, quantitative role, and claim boundary.
- `reference_registry.json`: machine-readable bibliographic and source-role records.
- `fluid_property_coefficients.csv`: inspectable coefficients used by the frozen property registry.
- `salt_skeleton_pair_registry.csv`: pair-specific physics and deployment evidence gates.
- `architecture_evidence_registry.csv`: architecture descriptors and common-dynamic-map eligibility.
- `skeleton_architecture_registry.csv` and `skeleton_property_registry.csv`: source-resolved skeleton evidence.
- `nanoparticle_*`: source-resolved nanoparticle composition, loading, temperature, and response evidence.
- `external_validation_source_registry.csv`: DLR source, access mode, calibration/holdout rule, and claim boundary.
- `dlr_weber_2025_figure_benchmark.csv`: executable figure-digitized benchmark.
- `dlr_weber_2025_figure_benchmark_PROVENANCE.md`: digitization method and permitted-use statement.

Absence is treated as a data gate. Missing chemistry, transport, cycle, hydraulic, structural, or uncertainty fields are not silently imputed. Cross-architecture effects are normalized only to matched within-source baselines.
