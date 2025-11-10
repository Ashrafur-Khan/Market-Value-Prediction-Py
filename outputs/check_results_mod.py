import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams.update({"figure.dpi": 140})

def _fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(round(x)):,}"

def _render_table_png(df: pd.DataFrame, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(10, 2 + 0.35*len(df)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                   cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    if title:
        ax.set_title(title, pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _make_paper_tables(preds: pd.DataFrame, out_dir: str):
    """
    Build two paper-style tables from test predictions:
      - Table A: RMSE/MAE/R2 for Model 3 (MLR, RF, GBM)
      - Table B: Hold-out RMSE only, laid out as Model 1/2/3 (we fill Model 3)
    """
    # Map your column names -> model labels
    model_cols = {
        "MLR": "pred_Linear_marketValue",
        "RF": "pred_RandomForest_marketValue",
        "GBM": "pred_GBM_marketValue",
    }
    if "Cost_eur" not in preds.columns:
        print("WARN: Cost_eur not found in predictions; cannot build paper tables.")
        return

    # Compute metrics vs. actual transfer fee on the test set
    rows = []
    rmse_only = {}
    for label, col in model_cols.items():
        if col not in preds.columns:
            continue
        df = preds[["Cost_eur", col]].copy()
        df["Cost_eur"] = pd.to_numeric(df["Cost_eur"], errors="coerce")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        if df.empty:
            continue
        y_true, y_pred = df["Cost_eur"].values, df[col].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        rows.append((label, rmse, mae, r2))
        rmse_only[label] = rmse

    # === Table A: Metrics (Model 3 block, like paper’s Table 4) ===
    # Order and numeric formatting match the paper’s look (commas, integers)
    order = ["MLR", "RF", "GBM"]
    tableA = pd.DataFrame(
        [(lab, *(next((rm, ma, r2) for (L, rm, ma, r2) in rows if L == lab)))
         for lab in order if any(L == lab for (L, *_ ) in rows)],
        columns=["Model", "RMSE", "MAE", "R²"]
    ).set_index("Model")

    # Format numbers with thousands separators (no decimals) to mimic the figures
    tableA_fmt = tableA.copy()
    tableA_fmt["RMSE"] = tableA["RMSE"].map(_fmt_int)
    tableA_fmt["MAE"]  = tableA["MAE"].map(_fmt_int)
    tableA_fmt["R²"]   = tableA["R²"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    # Save CSV/PNG
    a_csv = os.path.join(out_dir, "paper_table_metrics_model3.csv")
    a_png = os.path.join(out_dir, "paper_table_metrics_model3.png")
    tableA.to_csv(a_csv, float_format="%.6f")
    _render_table_png(tableA_fmt, a_png, title="Model 3 — RMSE / MAE / R²")

    # === Table B: Hold-out RMSE only (like paper’s Table 5) ===
    # Lay out rows as Model 1/2/3 and cols MLR/RF/GBM; fill Model 3 only.
    tableB = pd.DataFrame(index=["Model 1", "Model 2", "Model 3"], columns=order, dtype=float)
    for lab in order:
        if lab in rmse_only:
            tableB.loc["Model 3", lab] = rmse_only[lab]

    tableB_fmt = tableB.applymap(_fmt_int)
    b_csv = os.path.join(out_dir, "paper_table_holdout_rmse.csv")
    b_png = os.path.join(out_dir, "paper_table_holdout_rmse.png")
    tableB.to_csv(b_csv, float_format="%.6f")
    _render_table_png(tableB_fmt, b_png, title="Hold-out RMSE (€)")

    print("\nSaved paper-style tables:")
    print(" -", a_csv)
    print(" -", a_png)
    print(" -", b_csv)
    print(" -", b_png)

def main(outputs_dir: str):
    odir = outputs_dir
    #Load artifacts
    fc_path   = os.path.join(odir, "Factor_Conversion.csv")
    inds_path = os.path.join(odir, "Popularity_Indicators.csv")
    mdl_path  = os.path.join(odir, "modeling_table.csv")
    rmse_path = os.path.join(odir, "model_rmse.csv")
    fi_path   = os.path.join(odir, "feature_importances.csv")
    preds_path= os.path.join(odir, "test_predictions.csv")

    fc   = pd.read_csv(fc_path) if os.path.exists(fc_path) else None
    inds = pd.read_csv(inds_path) if os.path.exists(inds_path) else None
    mdl  = pd.read_csv(mdl_path) if os.path.exists(mdl_path) else None
    rmse = pd.read_csv(rmse_path) if os.path.exists(rmse_path) else None
    preds= pd.read_csv(preds_path) if os.path.exists(preds_path) else None

    print("=== Files found ===")
    for p in [fc_path, inds_path, mdl_path, rmse_path, fi_path, preds_path]:
        print(("OK  " if os.path.exists(p) else "MISS"), p)

    # ... (keep your existing summaries/plots here unchanged) ...

    # === NEW: build the two paper-style tables from predictions ===
    if preds is not None:
        _make_paper_tables(preds, odir)

if __name__ == '__main__':
    default_dir = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default=default_dir, help='Folder containing the CSV outputs')
    args = ap.parse_args()
    main(args.outputs)
