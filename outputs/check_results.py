"""Enhanced diagnostics and visualization suite for pipeline outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    plt.style.use("seaborn-v0_8")
except Exception:  # pragma: no cover - style availability varies by install
    pass

MIN_SEGMENT_SIZE = 5


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def ensure_figures_dir(outputs_dir: Path) -> Path:
    fig_dir = outputs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------------------------
# Visualization builders
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(mdl: pd.DataFrame, fig_dir: Path) -> None:
    indicator_cols = [c for c in ["PC1", "Mean", "Variance", "Min", "Max", "Median"] if c in mdl.columns]
    target_cols = [c for c in ["MarketValue_eur", "Cost_eur"] if c in mdl.columns]
    cols = indicator_cols + target_cols
    if not cols:
        return

    data = mdl[cols].apply(pd.to_numeric, errors="coerce")
    corr = data.corr()
    if corr.isna().all().all():
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Indicator / Target Correlations")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, fig_dir / "indicator_correlation_heatmap.png")


def plot_feature_importances(fi: pd.DataFrame, fig_dir: Path, top_n: int = 15) -> None:
    needed = {"feature", "importance", "model"}
    if not needed.issubset(fi.columns):
        return

    fi = fi.dropna(subset=["feature", "importance", "model"])
    if fi.empty:
        return

    for model, group in fi.groupby("model"):
        top = group.nlargest(top_n, "importance")
        if top.empty:
            continue
        top = top.sort_values("importance")

        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
        ax.barh(top["feature"], top["importance"], color="#1f77b4")
        ax.set_xlabel("Importance")
        ax.set_title(f"{model} Feature Importance (Top {len(top)})")
        save_figure(fig, fig_dir / f"feature_importance_{model}.png")


def plot_actual_vs_pred(preds: pd.DataFrame, actual_col: str, model_cols: list[str], fig_dir: Path) -> None:
    for col in model_cols:
        model_name = col.replace("pred_", "").replace("_marketValue", "")
        x = preds[actual_col]
        y = preds[col]
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue

        x_vals = x[mask]
        y_vals = y[mask]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_vals, y_vals, alpha=0.6, edgecolor="none", label="Predicted")

        lower = float(np.nanmin([x_vals.min(), y_vals.min()]))
        upper = float(np.nanmax([x_vals.max(), y_vals.max()]))
        ax.plot([lower, upper], [lower, upper], "--", color="grey", linewidth=1, label="Ideal")

        if mask.sum() >= 10:
            smooth = lowess(y_vals, x_vals, frac=0.3, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color="#d62728", linewidth=2, label="LOWESS")

        rmse = float(np.sqrt(np.mean((y_vals - x_vals) ** 2)))
        mae = float(np.mean(np.abs(y_vals - x_vals)))
        ax.text(
            0.05,
            0.95,
            f"RMSE: {rmse:,.0f}\nMAE: {mae:,.0f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Actual Transfer Fee (€)")
        ax.set_ylabel(f"{model_name} Prediction (€)")
        ax.set_title(f"{model_name}: Predicted vs Actual Fee")
        ax.legend()
        save_figure(fig, fig_dir / f"pred_vs_actual_{model_name}.png")


def plot_residual_scatter(pred_long: pd.DataFrame, fig_dir: Path) -> None:
    for model, group in pred_long.groupby("model"):
        data = group.dropna(subset=["prediction", "residual"])
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data["prediction"], data["residual"], alpha=0.6, edgecolor="none")
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Market Value (€)")
        ax.set_ylabel("Residual (Predicted - Actual)")
        ax.set_title(f"{model}: Residuals vs Predicted")
        save_figure(fig, fig_dir / f"residuals_vs_predicted_{model}.png")


def plot_residual_histograms(pred_long: pd.DataFrame, fig_dir: Path) -> None:
    for model, group in pred_long.groupby("model"):
        data = group["residual"].dropna()
        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(data, bins=30, color="#1f77b4", alpha=0.7)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (Predicted - Actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"{model}: Residual Distribution")
        save_figure(fig, fig_dir / f"residual_distribution_{model}.png")


def compute_segment_metrics(pred_long: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    if segment_col not in pred_long.columns:
        return pd.DataFrame()

    seg_df = pred_long.dropna(subset=[segment_col, "residual"])
    if seg_df.empty:
        return pd.DataFrame()

    metrics = seg_df.groupby(["model", segment_col]).agg(
        rmse=("residual", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mae=("residual", lambda x: float(np.mean(np.abs(x)))),
        count=("residual", "size"),
    ).reset_index()

    metrics = metrics[metrics["count"] >= MIN_SEGMENT_SIZE]
    return metrics


def plot_segment_metrics(metrics: pd.DataFrame, segment_col: str, fig_dir: Path) -> None:
    if metrics.empty:
        return

    csv_path = fig_dir / f"segment_metrics_{segment_col.lower()}.csv"
    metrics.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")

    for metric in ["rmse", "mae"]:
        pivot = metrics.pivot(index=segment_col, columns="model", values=metric)
        if pivot.empty:
            continue
        pivot = pivot.sort_values(by=list(pivot.columns), ascending=True)

        x_positions = np.arange(len(pivot.index))
        bar_width = 0.8 / max(1, len(pivot.columns))
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.index) * 0.7), 6))
        for idx, model in enumerate(pivot.columns):
            ax.bar(x_positions + idx * bar_width, pivot[model], width=bar_width, label=model)

        ax.set_xticks(x_positions + bar_width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by {segment_col}")
        ax.legend()
        save_figure(fig, fig_dir / f"{metric}_by_{segment_col.lower()}.png")


def spotlight_top_residuals(pred_long: pd.DataFrame, fig_dir: Path, top_n: int = 10) -> None:
    data = pred_long.dropna(subset=["residual"])
    if data.empty:
        return

    extremes = pd.concat(
        [data.nlargest(top_n, "residual"), data.nsmallest(top_n, "residual")],
        ignore_index=True,
    )
    if extremes.empty:
        return

    extremes["direction"] = np.where(extremes["residual"] > 0, "Overestimate", "Underestimate")
    extremes["abs_residual"] = extremes["residual"].abs()
    extremes = extremes.sort_values("residual")

    keep_cols = [
        "Player",
        "model",
        "residual",
        "prediction",
        "actual",
        "Position",
        "League",
        "Team",
        "Age",
    ]
    keep_cols = [c for c in keep_cols if c in extremes.columns]

    csv_path = fig_dir / "top_residuals.csv"
    extremes[keep_cols].to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")

    preview = extremes[keep_cols].copy()
    for col in ["residual", "prediction", "actual"]:
        if col in preview.columns:
            preview[col] = preview[col].round(0)
    print("\nTop residuals preview:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(preview.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, max(6, len(extremes) * 0.35)))
    colors = extremes["direction"].map({"Overestimate": "#d95f02", "Underestimate": "#1b9e77"})
    labels = extremes.apply(lambda row: f"{row.get('Player', 'Unknown')} ({row.get('model', '')})", axis=1)
    ax.barh(range(len(extremes)), extremes["residual"], color=colors)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(extremes)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Residual (Predicted - Actual)")
    ax.set_title("Top Residuals Spotlight")
    legend_handles = [
        plt.Line2D([0], [0], color="#d95f02", lw=6, label="Overestimate"),
        plt.Line2D([0], [0], color="#1b9e77", lw=6, label="Underestimate"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    save_figure(fig, fig_dir / "top_residuals_bar.png")


def build_predictions_long(preds: pd.DataFrame, actual_col: str, model_cols: list[str]) -> pd.DataFrame:
    id_candidates = ["Player", "Team", "Position", "Age", "League", "Nacionalidad", "Continent"]
    id_vars = [c for c in id_candidates if c in preds.columns]

    melted = preds.melt(
        id_vars=id_vars + [actual_col],
        value_vars=model_cols,
        var_name="model_col",
        value_name="prediction",
    )
    melted["model"] = (
        melted["model_col"].str.replace("pred_", "", regex=False).str.replace("_marketValue", "", regex=False)
    )
    melted["prediction"] = pd.to_numeric(melted["prediction"], errors="coerce")
    melted["actual"] = pd.to_numeric(melted[actual_col], errors="coerce")
    melted = melted.dropna(subset=["prediction", "actual"])
    melted["residual"] = melted["prediction"] - melted["actual"]
    melted.drop(columns=["model_col"], inplace=True)
    return melted

def plot_model_performance_bars(metrics: pd.DataFrame, fig_dir: Path) -> None:
    """Recreates the old generate_visuals() RMSE/MAE/CV bar charts."""
    required_cols = {"model", "rmse_test", "mae_test", "rmse_cv"}
    if not required_cols.issubset(metrics.columns):
        print("Skipping model performance bars: missing required columns.")
        return

    # RMSE Test
    df_rmse = metrics.sort_values("rmse_test")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df_rmse["model"], df_rmse["rmse_test"])
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title("Model Performance — Test RMSE")
    ax.grid(axis="x", alpha=0.4)
    save_figure(fig, fig_dir / "rmse_bar_chart.png")

    # MAE Test
    df_mae = metrics.sort_values("mae_test")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df_mae["model"], df_mae["mae_test"], color="#31a354")
    ax.set_xlabel("Test MAE (lower is better)")
    ax.set_title("Model Performance — Test MAE")
    ax.grid(axis="x", alpha=0.4)
    save_figure(fig, fig_dir / "mae_bar_chart.png")

    # CV RMSE
    df_cv = metrics.sort_values("rmse_cv")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df_cv["model"], df_cv["rmse_cv"], color="#de2d26")
    ax.set_xlabel("Cross-validation RMSE (lower is better)")
    ax.set_title("Model Performance — CV RMSE")
    ax.grid(axis="x", alpha=0.4)
    save_figure(fig, fig_dir / "cv_rmse_bar_chart.png")

# ---------------------------------------------------------------------------
# Main CLI flow
# ---------------------------------------------------------------------------

def main(outputs_dir: str | Path) -> None:
    outputs_path = Path(outputs_dir).resolve()
    fig_dir = ensure_figures_dir(outputs_path)

    paths = {
        "Factor_Conversion": outputs_path / "Factor_Conversion.csv",
        "Popularity_Indicators": outputs_path / "Popularity_Indicators.csv",
        "modeling_table": outputs_path / "modeling_table.csv",
        "model_rmse": outputs_path / "model_rmse.csv",
        "feature_importances": outputs_path / "feature_importances.csv",
        "test_predictions": outputs_path / "test_predictions.csv",
    }

    fc = load_csv(paths["Factor_Conversion"])
    inds = load_csv(paths["Popularity_Indicators"])
    mdl = load_csv(paths["modeling_table"])
    rmse = load_csv(paths["model_rmse"])
    fi = load_csv(paths["feature_importances"])
    preds = load_csv(paths["test_predictions"])

    print("=== Files found ===")
    for label, path in paths.items():
        print(("OK  " if path.exists() else "MISS"), path)

    if fc is not None:
        print(f"\nFactor_Conversion shape: {fc.shape}  (players x weeks)")
        row_stats = pd.DataFrame({
            "row_min": fc.min(axis=1),
            "row_max": fc.max(axis=1),
            "row_mean": fc.mean(axis=1),
        })
        print("Factor_Conversion row stats:\n", row_stats.describe().round(2))

    if inds is not None:
        print(f"\nPopularity_Indicators shape: {inds.shape}")
        print("Indicator columns:", inds.columns.tolist())

    if mdl is not None:
        print(f"\nModeling table shape: {mdl.shape}")
        pi_cols = ["PC1", "Mean", "Variance", "Min", "Max", "Median"]
        present = {c: (c in mdl.columns) for c in pi_cols}
        missing = {c: int(mdl[c].isna().sum()) if c in mdl.columns else None for c in pi_cols}
        print("Indicators present:", present)
        print("Indicator missing counts:", missing)

        if "Sold_flag" in mdl.columns:
            train = mdl[mdl["Sold_flag"].isna()].copy()
            test = mdl[mdl["Sold_flag"].eq(1.0)].copy()
        else:
            train = mdl.copy()
            test = mdl.copy()

        def s_corr(df: pd.DataFrame, col: str, target: str) -> float | None:
            if col not in df.columns or target not in df.columns:
                return None
            a = pd.to_numeric(df[col], errors="coerce")
            b = pd.to_numeric(df[target], errors="coerce")
            if a.notna().sum() == 0 or b.notna().sum() == 0:
                return None
            return float(a.corr(b))

        if not train.empty and "MarketValue_eur" in train.columns:
            print("\nCorrelations on TRAIN (vs MarketValue_eur):")
            for c in [col for col in pi_cols if col in train.columns]:
                corr_val = s_corr(train, c, "MarketValue_eur")
                if corr_val is not None:
                    print(f"  {c:>7}: {corr_val: .3f}")
        if not test.empty and "Cost_eur" in test.columns:
            print("\nCorrelations on TEST (vs Cost_eur):")
            for c in [col for col in pi_cols if col in test.columns]:
                corr_val = s_corr(test, c, "Cost_eur")
                if corr_val is not None:
                    print(f"  {c:>7}: {corr_val: .3f}")

        plot_correlation_heatmap(mdl, fig_dir)

    if rmse is not None:
        print("\n=== Model RMSE (lower is better) ===")
        print(rmse.sort_values("rmse_vs_fee"))
        # Add model performance bar charts (RMSE/MAE/CV)
        plot_model_performance_bars(rmse, fig_dir)

    if fi is not None:
        plot_feature_importances(fi, fig_dir)

    if preds is not None:
        actual_col = "Cost_eur" if "Cost_eur" in preds.columns else None
        model_cols = [c for c in preds.columns if c.startswith("pred_")]
        if actual_col is None or not model_cols:
            print("\nSkipping prediction diagnostics: required columns not found.")
        else:
            if mdl is not None and "Player" in mdl.columns:
                meta_cols = [c for c in ["Player", "League", "Nacionalidad", "Continent"] if c in mdl.columns]
                if meta_cols:
                    metadata = mdl[meta_cols].drop_duplicates("Player")
                    preds = preds.merge(metadata, on="Player", how="left")

            to_numeric(preds, [actual_col, "Age"] + model_cols)

            plot_actual_vs_pred(preds, actual_col, model_cols, fig_dir)

            pred_long = build_predictions_long(preds, actual_col, model_cols)
            if not pred_long.empty:
                plot_residual_scatter(pred_long, fig_dir)
                plot_residual_histograms(pred_long, fig_dir)

                for segment in ["Position", "League"]:
                    seg_metrics = compute_segment_metrics(pred_long, segment)
                    if not seg_metrics.empty:
                        print(f"\nSegment metrics by {segment} (count >= {MIN_SEGMENT_SIZE}):")
                        print(seg_metrics.sort_values(["model", segment]))
                        plot_segment_metrics(seg_metrics, segment, fig_dir)

                spotlight_top_residuals(pred_long, fig_dir)


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect and visualize pipeline outputs.")
    parser.add_argument("--outputs", default=default_dir, help="Folder containing the CSV outputs")
    args = parser.parse_args()
    main(args.outputs)
