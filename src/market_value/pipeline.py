"""High-level orchestration for the market value recreation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .config import PATHS, ProjectPaths
from .trends import build_factor_conversion, compute_indicators

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class ModelingData:
    #Container for the model-ready matrices.
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    features: List[str]
    test_view: pd.DataFrame


# Data preparation helpers 
def strip_accents(text: str) -> str:
    #Normalize accents for reliable joins across Excel sources
    if pd.isna(text):
        return text
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(char)
    ).strip()


def build_player_name_order_from_trends(trends_path: str) -> List[str]:
    #Replicate the player column ordering encoded in the Trends workbook
    xl = pd.ExcelFile(trends_path)

    def cols(df: pd.DataFrame, start: int, end: int) -> List[str]:
        return df.columns[start - 1 : end].tolist()

    names: List[str] = []

    df_fw = xl.parse("DELANTEROS")
    names += cols(df_fw, 2, 31)
    names += cols(df_fw, 32, 223)
    names += cols(df_fw, 224, 389)

    df_md = xl.parse("CENTROCAMPISTAS")
    names += cols(df_md, 2, 2)
    names += cols(df_md, 3, 89)
    names += cols(df_md, 90, 423)
    names += cols(df_md, 424, 446)

    df_df = xl.parse("DEFENSAS")
    names += cols(df_df, 2, 2)
    names += cols(df_df, 3, 85)
    names += cols(df_df, 86, 381)
    names += cols(df_df, 382, 598)

    return [strip_accents(name) for name in names]


def prepare_indicator_frame(paths: ProjectPaths) -> pd.DataFrame:
    #Generate factor conversion matrix and popularity indicators with player names
    factor_conversion = build_factor_conversion(
        str(paths.trends_workbook),
        out_csv_path=str(paths.output_dir / "Factor_Conversion.csv"),
    )
    indicators = compute_indicators(
        factor_conversion,
        indicator_start_col=1,
        exclude_first_col_for_pca=False,
        out_csv_path=str(paths.output_dir / "Popularity_Indicators.csv"),
    )

    player_order = build_player_name_order_from_trends(str(paths.trends_workbook))
    if len(player_order) != len(indicators):
        raise RuntimeError(
            f"Player name count ({len(player_order)}) != indicator rows ({len(indicators)})"
        )

    indicators["Player"] = player_order
    indicators["Player_key"] = indicators["Player"].apply(strip_accents)
    return indicators


def load_market_value_frames(paths: ProjectPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #Load the main market value table and the sold/test split workbook
    mv_main = pd.read_excel(paths.market_value_file, sheet_name="EuropaA_Wyscout_2018-19_FULL")
    mv_main["Player_key"] = mv_main["Player"].apply(strip_accents)

    mv_split = pd.read_excel(paths.market_value_file, sheet_name="Hoja2")
    mv_split = mv_split.rename(columns={"Player": "Player_mv2"})
    mv_split["Player_key"] = mv_split["Player_mv2"].apply(strip_accents)

    return mv_main, mv_split


def load_country_metadata(paths: ProjectPaths) -> pd.DataFrame:
    #Read supplementary geography data for enrichment
    countries = pd.read_excel(paths.countries_workbook, sheet_name="Hoja2")
    countries["Player_key"] = countries["Jugador"].apply(strip_accents)
    columns = ["Player_key", "League", "Nacionalidad", "Continent"]
    return countries[columns].drop_duplicates("Player_key")


def assemble_model_base(
    mv_main: pd.DataFrame,
    mv_split: pd.DataFrame,
    indicators: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    #Combine all sources into the modeling table used downstream
    base = mv_main.merge(countries, on="Player_key", how="left")
    base = base.merge(
        indicators.drop_duplicates("Player_key"),
        on="Player_key",
        how="left",
        suffixes=("", "_pi"),
    )
    base = base.merge(mv_split[["Player_key", "Sold_flag"]], on="Player_key", how="left")
    return base


def create_model_matrices(base: pd.DataFrame) -> ModelingData:
    #Split the modeling table into train/test matrices and series
    num_cols = base.select_dtypes(include=["number"]).columns.tolist()
    target_mv = "MarketValue_eur"
    true_fee = "Cost_eur"
    drop_cols = {target_mv, true_fee, "Sold_flag", "Contract_years"}
    feature_cols = [col for col in num_cols if col not in drop_cols]

    train = base[base["Sold_flag"].isna()].copy()
    test = base[base["Sold_flag"].eq(1.0)].copy()

    train = train[train[target_mv].notna()]
    test = test[test[true_fee].notna()]

    X_train = train[feature_cols].fillna(0.0)
    y_train = train[target_mv].astype(float)
    X_test = test[feature_cols].fillna(0.0)
    y_test = test[true_fee].astype(float)

    test_view = test[["Player", "Team", "Position", "Age", true_fee]].copy()

    return ModelingData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        features=feature_cols,
        test_view=test_view,
    )


# --- Modeling helpers ---------------------------------------------------------

def train_models(data: ModelingData, n_splits: int = 5) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    """
    Train and evaluate multiple regression models using K-Fold cross-validation and log-transformed target.
    Returns: (metrics DataFrame, predictions dict, fitted models dict)
    """
    # --- Step 1: Apply log-transform to target to stabilize scale ---
    y_train_log = np.log1p(data.y_train)
    y_test_log = np.log1p(data.y_test)

    # --- Step 2: Define models ---
    models: Dict[str, object] = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.001, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=2000, random_state=42, n_jobs=-1),
        "GBM": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=1000, random_state=42, n_jobs=-1, tree_method="hist"),
        "LightGBM": LGBMRegressor(n_estimators=1000, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.05, loss_function="RMSE", 
                                      random_seed=42, verbose=False, allow_writing_files=False)
    }

    results: List[Dict[str, float]] = []
    predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, object] = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- Step 3: Cross-validation & model fitting ---
    for name, model in models.items():
        print(f"\nTraining {name} with {n_splits}-Fold CV...")

        cv_scores = cross_val_score(
            model,
            data.X_train,
            y_train_log,  # use log target
            scoring="neg_root_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        mean_rmse_cv = -np.mean(cv_scores)

        model.fit(data.X_train, y_train_log)
        preds_log = model.predict(data.X_test)
        preds = np.expm1(preds_log)  # inverse of log1p for final predictions
        rmse_test = float(mean_squared_error(data.y_test, preds))

        results.append({
            "model": name,
            "rmse_cv": mean_rmse_cv,
            "rmse_test": rmse_test
        })

        predictions[name] = preds
        fitted_models[name] = model

    # --- Step 4: Combine and sort results ---
    metrics = pd.DataFrame(results).sort_values("rmse_cv")
    print("\n=== RMSE Results (lower is better) ===")
    print(metrics)

    return metrics, predictions, fitted_models



def compute_feature_importances(
    fitted_models: Dict[str, object],
    feature_names: List[str],
) -> pd.DataFrame:
    #Aggregate feature importances for models that expose them
    frames: List[pd.DataFrame] = []
    for name, model in fitted_models.items():
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            fi["model"] = name
            frames.append(fi)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values(["model", "importance"], ascending=[True, False])


# Public pipeline API 

def run_pipeline(paths: ProjectPaths | None = None) -> pd.DataFrame:
    #Execute the full pipeline and return the RMSE leaderboard.
    paths = paths or PATHS
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    indicators = prepare_indicator_frame(paths)
    mv_main, mv_split = load_market_value_frames(paths)
    countries = load_country_metadata(paths)
    base = assemble_model_base(mv_main, mv_split, indicators, countries)
    modeling_data = create_model_matrices(base)

    metrics, predictions, fitted_models = train_models(modeling_data)

    metrics.to_csv(paths.output_dir / "model_rmse.csv", index=False)
    base.to_csv(paths.output_dir / "modeling_table.csv", index=False)

    pred_frame = modeling_data.test_view.copy()
    for name, values in predictions.items():
        pred_frame[f"pred_{name}_marketValue"] = values
    pred_frame.to_csv(paths.output_dir / "test_predictions.csv", index=False)

    importances = compute_feature_importances(fitted_models, modeling_data.features)
    if not importances.empty:
        importances.to_csv(paths.output_dir / "feature_importances.csv", index=False)

    print("=== RMSE vs actual fee (lower is better) ===")
    print(metrics.to_string(index=False))
    print(f"\nOutputs written to: {paths.output_dir}")

    return metrics

def run_position_specific_pipeline(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or PATHS
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    indicators = prepare_indicator_frame(paths)
    mv_main, mv_split = load_market_value_frames(paths)
    countries = load_country_metadata(paths)
    base = assemble_model_base(mv_main, mv_split, indicators, countries)

    base["Position"] = base["Position"].astype(str).fillna("Unknown").str.upper()


    position_groups = {
    "FORWARDS": [
        "FW", "CF", "ST", "LW", "RW", "ATTACK", "FORWARD", "WINGER", "DELANTERO"
    ],
    "MIDFIELDERS": [
        "MF", "CM", "CAM", "CDM", "MIDFIELD", "PLAYMAKER", "CENTROCAMPISTA"
    ],
    "DEFENDERS": [
        "DF", "CB", "LB", "RB", "CB", "DEFENDER", "FULL BACK", "DEFENSA"
    ],
    "GOALKEEPERS": [
        "GK", "GOALKEEPER", "KEEPER", "PORTERO"
    ]
}


    def get_position_group(pos: str) -> str:
        for group_name, aliases in position_groups.items():
            if any(alias in pos for alias in aliases):
                return group_name
        return "OTHER"

    base["PositionGroup"] = base["Position"].apply(get_position_group)

    all_metrics = []
    all_predictions = []

    for group_name, group_df in base.groupby("PositionGroup"):
        print(f"\n=== Training models for {group_name} ({len(group_df)} players) ===")

        try:
            modeling_data = create_model_matrices(group_df)
            if len(modeling_data.X_train) < 20:
                print(f"Skipping {group_name}: not enough training samples.")
                continue

            metrics, predictions, fitted_models = train_models(modeling_data)
            metrics["PositionGroup"] = group_name
            all_metrics.append(metrics)

            pred_frame = modeling_data.test_view.copy()
            for name, values in predictions.items():
                pred_frame[f"pred_{name}_marketValue"] = values
            pred_frame["PositionGroup"] = group_name
            all_predictions.append(pred_frame)

            importances = compute_feature_importances(fitted_models, modeling_data.features)
            if not importances.empty:
                importances["PositionGroup"] = group_name
                importances.to_csv(paths.output_dir / f"feature_importances_{group_name}.csv", index=False)

            metrics.to_csv(paths.output_dir / f"model_rmse_{group_name}.csv", index=False)
            pred_frame.to_csv(paths.output_dir / f"test_predictions_{group_name}.csv", index=False)

        except Exception as e:
            print(f"Skipping {group_name} due to error: {e}")

    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(paths.output_dir / "all_position_metrics.csv", index=False)
        print("\n=== Combined RMSE results across all positions ===")
        print(all_metrics_df.groupby("PositionGroup")["rmse_test"].mean().sort_values())
    else:
        all_metrics_df = pd.DataFrame()
        print("\nNo valid models were trained.")

    if all_predictions:
        all_preds_df = pd.concat(all_predictions, ignore_index=True)
        all_preds_df.to_csv(paths.output_dir / "all_position_predictions.csv", index=False)
    else:
        all_preds_df = pd.DataFrame()

    print(f"\nOutputs written to: {paths.output_dir}")
    return all_metrics_df
