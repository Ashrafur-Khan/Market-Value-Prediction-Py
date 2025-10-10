#Utilities for translating the R trends feature engineering into Python
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class LayerRanges:
    # 1-based inclusive column indices (R-style)
    reference: Optional[Tuple[int, int]]
    fl: Optional[Tuple[int, int]]
    sl: Optional[Tuple[int, int]]
    tl: Optional[Tuple[int, int]]


@dataclass
class LayerScales:
    reference_scale: float
    fl_scale: float
    sl_scale: float
    tl_scale: float


@dataclass
class SheetSpec:
    name: str
    ranges: LayerRanges
    scales: LayerScales


def _slice_1based_inclusive(df: pd.DataFrame, start_end: Tuple[int, int]) -> pd.DataFrame:
    #Slice columns using 1-based inclusive indexing (like R)
    start, end = start_end
    return df.iloc[:, start - 1 : end]


def _apply_layer(df: pd.DataFrame, rng: Optional[Tuple[int, int]], scale: float) -> pd.DataFrame:
    if rng is None:
        return pd.DataFrame(index=df.index)
    block = _slice_1based_inclusive(df, rng).copy()
    for col in block.columns:
        if pd.api.types.is_numeric_dtype(block[col]):
            block[col] = block[col].astype(float) * scale
        else:
            block[col] = pd.to_numeric(block[col], errors="coerce") * scale
    return block


def build_factor_conversion(
    excel_path: str,
    out_csv_path: Optional[str] = None,
    sheet_specs: Optional[List[SheetSpec]] = None,
) -> pd.DataFrame:
    #Reproduce the R pipeline: build layered/scaled matrix and transpose it
    if sheet_specs is None:
        sheet_specs = [
            SheetSpec(
                name="DELANTEROS",
                ranges=LayerRanges(reference=None, fl=(2, 31), sl=(32, 223), tl=(224, 389)),
                scales=LayerScales(reference_scale=1.0, fl_scale=1.0, sl_scale=0.04, tl_scale=0.0016),
            ),
            SheetSpec(
                name="CENTROCAMPISTAS",
                ranges=LayerRanges(reference=(2, 2), fl=(3, 89), sl=(90, 423), tl=(424, 446)),
                scales=LayerScales(reference_scale=1.0, fl_scale=0.15, sl_scale=0.0045, tl_scale=0.000135),
            ),
            SheetSpec(
                name="DEFENSAS",
                ranges=LayerRanges(reference=(2, 2), fl=(3, 85), sl=(86, 381), tl=(382, 598)),
                scales=LayerScales(reference_scale=1.0, fl_scale=0.14, sl_scale=0.0112, tl_scale=0.00108),
            ),
        ]

    xl = pd.ExcelFile(excel_path)
    pieces = []

    for spec in sheet_specs:
        if spec.name not in xl.sheet_names:
            raise ValueError(f"Sheet '{spec.name}' not found. Available: {xl.sheet_names}")
        df = xl.parse(spec.name)

        ref_block = _apply_layer(df, spec.ranges.reference, spec.scales.reference_scale)
        fl_block = _apply_layer(df, spec.ranges.fl, spec.scales.fl_scale)
        sl_block = _apply_layer(df, spec.ranges.sl, spec.scales.sl_scale)
        tl_block = _apply_layer(df, spec.ranges.tl, spec.scales.tl_scale)

        blocks = []
        if spec.name == "DELANTEROS":
            blocks.extend([fl_block, sl_block, tl_block])
        else:
            blocks.extend([ref_block, fl_block, sl_block, tl_block])

        sheet_matrix = pd.concat(blocks, axis=1)
        pieces.append(sheet_matrix)

    factor_conversion = pd.concat(pieces, axis=1)
    factor_conversion_T = factor_conversion.T  # match t(...) in R

    if out_csv_path:
        factor_conversion_T.to_csv(out_csv_path, index=False)

    return factor_conversion_T


def compute_indicators(
    matrix_df: pd.DataFrame,
    indicator_start_col: int = 1,
    exclude_first_col_for_pca: bool = False,
    out_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    #Compute PCA-based and summary indicators across weekly columns
    numeric_df = matrix_df.copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    if exclude_first_col_for_pca and numeric_df.shape[1] > 1:
        pca_input = numeric_df.iloc[:, 1:].to_numpy()
    else:
        pca_input = numeric_df.to_numpy()

    col_means = np.nanmean(pca_input, axis=0, keepdims=True)
    inds = np.where(np.isnan(pca_input))
    pca_input[inds] = np.take_along_axis(col_means, np.expand_dims(inds[1], 0), axis=1)[0]

    pc1 = PCA(n_components=1, random_state=42).fit_transform(pca_input).ravel()

    start_idx = max(0, indicator_start_col - 1)
    sub = numeric_df.iloc[:, start_idx:]

    indicators = pd.DataFrame(
        {
            "PC1": pc1,
            "Mean": sub.mean(axis=1, skipna=True),
            "Variance": sub.var(axis=1, ddof=1, skipna=True),
            "Min": sub.min(axis=1, skipna=True),
            "Max": sub.max(axis=1, skipna=True),
            "Median": sub.median(axis=1, skipna=True),
        }
    )

    rolling_windows = [3, 5]
    for w in rolling_windows:
        indicators[f"rolling_mean_{w}w"] = sub.rolling(window = w, axis = 1, min_periods = 1).mean().mean(axis = 1)
        indicators[f"rolling_std_{w}w"] = sub.rolling(window = w, axis = 1, min_periods = 1).std().mean(axis = 1)
        
    if out_csv_path:
        indicators.to_csv(out_csv_path, index=False)

    return indicators
