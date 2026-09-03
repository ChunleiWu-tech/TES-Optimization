# Cross-scale co-design of molten-salt packed-bed thermal storage

This repository is the GitHub-ready, reproducible V18 release accompanying a study of molten-salt packed-bed thermal-energy-storage co-design. It asks a specific design question: when does a fluid-property difference remain sufficiently large, after propagation through transient packed-bed physics and integer module sizing, to change the selected modular configuration?

The repository separates three evidence layers:

1. **Upstream model.** A conservative, cycle-coupled local-thermal-nonequilibrium packed-bed model evaluates the storage module, including an outlet-temperature qualification criterion.
2. **Service and decision analysis.** Prescribed power–energy requirements are closed with integer module counts. The V18 analysis tests whether a continuous material difference crosses the governing constraint and the next whole-module threshold.
3. **Postprocessing.** The figure workflow regenerates six main figures and nine supplementary figures as vector PDF files from the validated V18 result tables.

The terms *module* and *storage unit* refer to one physical packed bed. A *modular system* is the integer number of identical parallel modules required to meet the specified service requirement.

## Contents

```text
upstream_package/
  tespub/                 model and analysis source code
  config/                 frozen V16 and V18 study definitions
  data/                   property, validation, and evidence registries
  results_v16/            validated baseline tables required for reproduction
  results_v18/            validated integer-reclosure and module-count threshold tables
  scripts/                staged analysis and audit entry points
  tests/                  scientific regression tests
postprocessing_package/
  scripts/                publication-figure builder and fail-closed audit
  FIGURE_CAPTIONS_V18.md  complete captions for the 15 composite figures
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

The validated V16/V18 tables are included so figures can be regenerated immediately:

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

Before making a public repository, complete the author, title, DOI, and licence fields described in [CITATION.md](CITATION.md) and [LICENSE_NOTICE.md](LICENSE_NOTICE.md). No open-source licence is granted by this draft repository.
