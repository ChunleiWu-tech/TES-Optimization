# Reproducibility guide

## Software environment

Use Python 3.10 or later. The exact runtime packages for the released analysis are listed in `requirements.txt`. Install the upstream package in editable mode from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .\upstream_package
```

## Included validated state

`upstream_package/results_v16/` is the frozen baseline and `upstream_package/results_v18/` is the validated design-consequence extension. Together they allow exact regeneration of the figure data without repeating the long transient computation. Do not alter either directory before running the figure workflow.

The included state contains 3,840 unit-scale cases, 7,680 service cases, 15,360 matched fluid-pair comparisons, and the V18 integer-reclosure and continuous-closure records. The release audit requires zero pairwise rank reversals, 218 baseline integer ties, and 52 objective-resolution alternatives after the V18 decision-boundary analysis.

## Verification without recomputation

```powershell
python .\upstream_package\scripts\audit_v16_results.py
python -m unittest discover -s .\upstream_package\tests -p "test_*.py"
$env:TES_FIGURE_OUTPUT_ROOT = (Resolve-Path .\postprocessing_package).Path
python .\postprocessing_package\scripts\build_v18_figures.py
python .\postprocessing_package\scripts\audit_v18_figures.py
```

The figure audit requires six main and nine supplementary one-page vector PDFs, Arial text, external panel letters, no embedded panel titles, complete captions, and no visible missing-value or release-version tokens.

## Full staged recomputation

The baseline computation is long-running. Run it only in a clean working copy because `run_v16.py` replaces `results_v16/` during controlled promotion. The V18 stages must be performed in this order:

```powershell
# Baseline rerun (long-running; use a clean copy)
python .\upstream_package\run_v16.py --no-archive

# V18 independent integer reclosure and corrected service closure
python .\upstream_package\scripts\run_v18_integer_reclosure_audit.py --workers 8
python .\upstream_package\scripts\build_v18_corrected_service.py --workers 8

# Rebuild tables that depend on integer module count
python .\upstream_package\scripts\rebuild_v18_service_dependents.py
python .\upstream_package\scripts\rebuild_v18_monotonicity_probe.py --workers 8

# Continuous closure, pairwise design-consequence analysis, and release audit
python .\upstream_package\scripts\run_v18_design_consequence.py --mode full --workers 8
python .\upstream_package\scripts\finalize_v18_release.py
```

Use no more worker processes than physical CPU cores and reduce `--workers` if memory pressure occurs. Runtime checkpoints and release records are deliberately Git-ignored so that local recovery state is never committed.

## Regenerate figures

```powershell
$env:TES_FIGURE_OUTPUT_ROOT = (Resolve-Path .\postprocessing_package).Path
python .\postprocessing_package\scripts\build_v18_figures.py
python .\postprocessing_package\scripts\audit_v18_figures.py
```

The builder imports `build_v16_figures.py` as a shared plotting library and applies the V18 data sources where the design-consequence analysis changes. Figure 5 uses the continuous closure, pairwise design-consequence, and integer-reclosure audit; source-specific architecture evidence remains in Supplementary Figure S8.
