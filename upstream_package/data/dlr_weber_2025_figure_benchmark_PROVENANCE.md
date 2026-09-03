# DLR Weber 2025 figure-benchmark provenance

## Source

Weber et al., *Experimental dataset of fluid flow and heat transfer in a shallow packed bed at low Reynolds numbers*, Data in Brief 61 (2025) 111743, DOI 10.1016/j.dib.2025.111743. Associated Mendeley Data V2: DOI 10.17632/3pp86gdvh4.2.

The 2026 corrigendum is Data in Brief 68 (2026) 113089, DOI 10.1016/j.dib.2026.113089. It corrects bibliographic information and does not revise the measurement repository, experimental values, or the plotted temperature trace digitized for this benchmark.

## Access mode

The article PDF is included under CC BY 4.0. The raw Mendeley archive is not included in this release. `dlr_weber_2025_figure_benchmark.csv` was reconstructed from the published temperature curves and therefore carries `digitisation_allowance_K = 5.0` on every row.

## Observable construction

The CSV contains:

- time from the start of the published heating-cooling trace;
- reconstructed area-weighted outlet/upper-bed temperature;
- lower and upper radial measurement envelopes;
- reconstructed inlet/boundary temperature.

The benchmark is deliberately labelled figure-level. It is not represented as a raw sensor export and is not suitable for re-estimating experimental uncertainty.

## Calibration and holdout

- Heating segment: calibrate one volumetric wall-loss coefficient only.
- Cooling segment: retain as temporal holdout.
- No fitted Nusselt multiplier, particle-conductivity multiplier, axial-dispersion multiplier, or cooling-specific parameter is used.

The cooling mismatch and radial-envelope comparison are reported as model-form evidence. They are not tuned away.

## Permitted use

The benchmark supports validation of the conservative one-dimensional LTNE model class and demonstrates its radial-flow limitation. It does not validate molten-salt chemistry, salt-skeleton compatibility, corrosion, long-cycle stability, or a deployable thermal-storage system.
