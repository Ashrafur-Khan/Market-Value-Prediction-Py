# Market Value Prediction Project Overview

This document introduces the structure of the project, highlights the most
important components, and suggests next steps for digging deeper into the
codebase and related concepts.

## Repository layout

The repository is intentionally small and is organized around a single data
pipeline that recreates an academic project for estimating association football
(soccer) player market values. Key top-level directories are:

- `src/`: Python source code packaged under `market_value/`, which builds
  engineered features from raw Excel exports and trains simple baseline models.
- `data/`: Input Excel workbooks. These are required to rerun the end-to-end
  pipeline. (Large files are typically tracked with Git LFS in the original
  project; in this kata the data is included in-repo for convenience.)
- `outputs/`: Generated artifacts such as feature tables, indicator summaries,
  model metrics, and prediction exports. Running the pipeline will populate
  this directory.
- `requirements.txt`: Python dependencies needed to reproduce the analysis.

The project does not use a package manager or CLI entry point—scripts are run
with `python` directly from the repository root.

## Core modules

### `src/market_value/trends.py`

This module translates the R workflow that layers Google Trends time series by
player position into Python data classes. `LayerRanges` and `LayerScales`
describe which spreadsheet columns correspond to each positional layer and the
scaling applied to them. `SheetSpec` packages those ranges + scales for a single
Excel sheet.

Key functions:

- `build_factor_conversion(...)` reads the Trends workbook, slices the desired
  column blocks for each position, rescales them, and concatenates everything
  into one large transposed matrix. Optionally, it writes the engineered matrix
  to CSV for inspection.【F:src/market_value/trends.py†L52-L105】
- `compute_indicators(...)` applies a Principal Component Analysis (PCA) on the
  layered matrix to extract a first principal component (PC1) indicator and
  computes summary statistics (mean, variance, min, max, median) across the
  weekly trend columns. Missing values are imputed with column means prior to
  running PCA.【F:src/market_value/trends.py†L108-L147】

### `src/market_value/pipeline.py`

`pipeline.py` groups the higher-level orchestration into reusable functions:

1. Build the layered/scaled trend matrix and compute popularity indicators.
2. Reconstruct the player ordering from the Trends workbook to align rows with
   market value data and enrich indicators with clean player keys.【F:src/market_value/pipeline.py†L42-L92】
3. Load the market value dataset, sold/test split, and supplementary country
   metadata used for enrichment.【F:src/market_value/pipeline.py†L95-L130】
4. Assemble the modeling table and derive train/test matrices for the models.
5. Fit three baseline regressors (Linear Regression, Random Forest, Gradient
   Boosting) to predict market value and evaluate them on sold players using
   RMSE against actual transfer fees. The results are written to CSV along with
   model predictions and optional feature importances.【F:src/market_value/pipeline.py†L133-L238】

### `src/recreate_project.py`

This thin compatibility wrapper exposes the original CLI. Running
`python src/recreate_project.py` from the repository root triggers
`market_value.pipeline.run_pipeline`, creates the derived datasets under
`outputs/`, and prints the evaluation summary.【F:src/recreate_project.py†L1-L12】

## Important implementation details

- **1-based Excel indexing**: The helper `_slice_1based_inclusive` emulates R’s
  indexing convention to match the original study exactly. When adapting the
  pipeline, double-check any changes to column ranges to ensure they align with
  the intended spreadsheet layout.【F:src/market_value/trends.py†L30-L44】
- **Data cleaning with accent stripping**: Player names are normalized via
  Unicode accent stripping to construct stable join keys across disparate
  spreadsheets. Any new data sources should undergo the same normalization to
  avoid mismatches.【F:src/market_value/pipeline.py†L27-L52】
- **Model feature selection**: The pipeline selects all numeric features except
  a handful of leakage-prone columns. When introducing new features or targets,
  review the `drop_cols` list to keep training labels isolated.【F:src/market_value/pipeline.py†L140-L166】
