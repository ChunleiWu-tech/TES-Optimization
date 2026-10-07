# Cross-scale transient co-design reveals when material differences change modular thermal-storage architecture

This repository accompanies a study of transient, cross-scale co-design for molten-salt packed-bed thermal energy storage. It addresses a specific scientific question: when does a thermophysical-property difference remain sufficiently large after liquid displacement, transient fluid–solid heat transfer, flow redistribution, competing service constraints, and integer module sizing to alter the modular storage architecture?

The repository separates three evidence layers:

1. **Upstream model.** A conservative, cycle-coupled local-thermal-nonequilibrium packed-bed model evaluates the storage module, including an outlet-temperature qualification criterion.
2. **Service and decision analysis.** Prescribed power–energy requirements are closed with integer module counts. The analysis determines whether a continuous material difference changes the active energy or rated-power constraint and exceeds an integer module-count threshold.
3. **Statistical design and postprocessing.** The frozen deterministic results are extended with base-geometry cluster-bootstrap intervals, scale-by-scale attenuation decomposition, adversarial integer-boundary stresses, and Pareto-set stability. The figure workflow regenerates six main figures and ten supplementary figures as vector PDF files.

The terms *module* and *storage unit* refer to one physical packed bed. A *modular system* is the integer number of identical parallel modules required to meet the specified service requirement.

## Contents

```text
upstream_package/
  tespub/                 model and analysis source code
  config/                 frozen V16 and V18 study definitions
  data/                   property, validation, and evidence registries
  results_v16/            validated baseline tables required for reproduction
  results_v18/            validated integer-reclosure and module-count threshold tables
  results_v19/            statistical-design and decision-boundary stability tables
  scripts/                staged analysis and audit entry points
  tests/                  scientific regression tests
postprocessing_package/
  scripts/                publication-figure builder and fail-closed audit
  FIGURE_CAPTIONS_V18.md  complete captions for the 16 composite figures
docs/                     workflow, reproducibility, and GitHub upload notes
```

The repository intentionally excludes the submission Word files, final publication PDFs, generated figure PDFs, runtime logs, cached binaries, and the locally stored copy of a cited journal article. Bibliographic and provenance records remain in `upstream_package/data/`.

## Quick start

Create an isolated Python environment (Python 3.10 or later), install the locked dependencies, install the upstream package, and run the retained scientific tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .\upstream_package
python -m unittest discover -s .\upstream_package\tests -p "test_*.py"
```

The validated V16/V18 tables and the V18.1 statistical extension are included so figures can be regenerated immediately:

```powershell
$env:TES_FIGURE_OUTPUT_ROOT = (Resolve-Path .\postprocessing_package).Path
python .\postprocessing_package\scripts\build_v18_figures.py
python .\postprocessing_package\scripts\audit_v18_figures.py
```

Generated PDFs are written below `postprocessing_package/figures/`; they are intentionally ignored by Git. Full staged rerun instructions and expected release gates are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Scope and claim boundary

The repository provides a physics-based comparison of salt-property effects within the registered model scope. It does **not** establish deployment readiness, corrosion compatibility, wetting behaviour, long-duration materials stability, or economic viability. Architecture and particle evidence is retained on its original experimental basis; cross-study material rankings are not inferred where matched evidence is unavailable.

## Upload to GitHub

This release is ready for Git CLI upload. Its validated CSV data are approximately 307 MiB in total, and several files exceed the browser uploader's practical limit. Use the Git command-line workflow in [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md), rather than drag-and-drop upload in a browser.

## Citation and licence

The public repository is https://github.com/ChunleiWu-tech/TES-Optimization. Cite the repository together with the commit hash used for analysis.
