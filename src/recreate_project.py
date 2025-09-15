# src/recreate_project.py

import sys, os
sys.path.append(os.path.dirname(__file__))  

import os, unicodedata
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ccf_from_trends import build_factor_conversion, compute_indicators

DATA_DIR = os.path.join("..", "data")
OUT_DIR  = os.path.join("..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MV_FILE   = os.path.join(DATA_DIR, "MarketValueData_English.xlsx")
TRENDS_XL = os.path.join(DATA_DIR, "Trends_FactorConversion.xlsx")
COUNTRIES = os.path.join(DATA_DIR, "Countries.xlsx")

def strip_accents(s: str) -> str:
    if pd.isna(s):
        return s
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    ).strip()

def build_player_name_order_from_trends(trends_path: str) -> list[str]:
    """
    Reconstruct the *exact* player-column order used in the layered matrix,
    matching the R cbind() logic and our Python translation.
    """
    xl = pd.ExcelFile(trends_path)

    def cols(df, a, b):  # 1-based inclusive -> 0-based
        return df.columns[a-1:b].tolist()

    names = []

    #DELANTEROS (forwards): FL 2:31, SL 32:223, TL 224:389
    df_fw = xl.parse("DELANTEROS")
    names += cols(df_fw, 2, 31)     # FL
    names += cols(df_fw, 32, 223)   # SL
    names += cols(df_fw, 224, 389)  # TL

    #CENTROCAMPISTAS (midfielders): REF 2, FL 3:89, SL 90:423, TL 424:446
    df_md = xl.parse("CENTROCAMPISTAS")
    names += cols(df_md, 2, 2)      # REFERENCE_MIDFIELDER
    names += cols(df_md, 3, 89)     # FL
    names += cols(df_md, 90, 423)   # SL
    names += cols(df_md, 424, 446)  # TL

    #DEFENSAS (defenders): REF 2, FL 3:85, SL 86:381, TL 382:598
    df_df = xl.parse("DEFENSAS")
    names += cols(df_df, 2, 2)      # REFERENCE_DEFENDER
    names += cols(df_df, 3, 85)     # FL
    names += cols(df_df, 86, 381)   # SL
    names += cols(df_df, 382, 598)  # TL

    #Normalize for merging keys
    return [strip_accents(n) for n in names]

def main():
    #1) Build layered & scaled matrix and compute popularity indicators
    fc = build_factor_conversion(TRENDS_XL,
                                 out_csv_path=os.path.join(OUT_DIR, "Factor_Conversion.csv"))
    #Indicators across ALL weekly columns
    inds = compute_indicators(fc, indicator_start_col=1, exclude_first_col_for_pca=False,
                              out_csv_path=os.path.join(OUT_DIR, "Popularity_Indicators.csv"))

    # Attach player names in the correct order (row order of fc == concat order above)
    trends_player_order = build_player_name_order_from_trends(TRENDS_XL)
    if len(trends_player_order) != len(inds):
        raise RuntimeError(f"Player name count ({len(trends_player_order)}) != indicator rows ({len(inds)})")

    inds["Player"] = trends_player_order
    inds["Player_key"] = inds["Player"].apply(strip_accents)

    # 2) Load MarketValue main sheet + sold/test sheet; create keys
    mv_main = pd.read_excel(MV_FILE, sheet_name="EuropaA_Wyscout_2018-19_FULL")
    mv_main["Player_key"] = mv_main["Player"].apply(strip_accents)

    mv_split = pd.read_excel(MV_FILE, sheet_name="Hoja2")
    mv_split.rename(columns={"Player": "Player_mv2"}, inplace=True)
    mv_split["Player_key"] = mv_split["Player_mv2"].apply(strip_accents)

    # 3) Countries enrichment (League/Nationality/Continent)
    ctry = pd.read_excel(COUNTRIES, sheet_name="Hoja2")
    # 'Jugador' = player name in Countries.xlsx
    ctry["Player_key"] = ctry["Jugador"].apply(strip_accents)
    ctry_slim = ctry[["Player_key", "League", "Nacionalidad", "Continent"]].drop_duplicates("Player_key")

    # 4) Merge master table
    base = mv_main.merge(ctry_slim, on="Player_key", how="left")
    base = base.merge(inds.drop_duplicates("Player_key"),
                      on="Player_key", how="left",
                      suffixes=("", "_pi"))

    # 5) Train/test split:
    #    - Train on players NOT sold (Sold_flag is NaN) with target = MarketValue_eur.
    #    - Test/evaluate on players sold (Sold_flag == 1) vs actual transfer fee = Cost_eur (present in main).
    base = base.merge(mv_split[["Player_key", "Sold_flag"]], on="Player_key", how="left")

    # features: all numeric except the targets/leakage fields
    num_cols = base.select_dtypes(include=["number"]).columns.tolist()
    target_mv = "MarketValue_eur"
    true_fee  = "Cost_eur"

    drop_cols = {target_mv, true_fee, "Sold_flag", "Contract_years"}  # drop targets + avoid leakage-ish cols if desired
    X_cols = [c for c in num_cols if c not in drop_cols]

    train = base[base["Sold_flag"].isna()].copy()
    test  = base[base["Sold_flag"].eq(1.0)].copy()

    # keep rows with targets
    train = train[train[target_mv].notna()]
    test  = test[test[true_fee].notna()]

    X_train = train[X_cols].fillna(0.0)
    y_train = train[target_mv].astype(float)

    X_test  = test[X_cols].fillna(0.0)
    y_test  = test[true_fee].astype(float)

    # 6) Models: Linear Regression, Random Forest, GBM
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=5000, random_state=42, n_jobs=-1),
        "GBM": GradientBoostingRegressor(random_state=42),
    }

    results = []
    preds_out = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        #rmse = mean_squared_error(y_test, preds, squared=False)

        try:
            rmse = float(mean_squared_error(y_test, preds, squared=False))  # newer sklearn
        except TypeError:
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))        # older sklearn

        results.append({"model": name, "rmse_vs_fee": float(rmse)})
        preds_out[name] = preds

    res_df = pd.DataFrame(results).sort_values("rmse_vs_fee")
    res_df.to_csv(os.path.join(OUT_DIR, "model_rmse.csv"), index=False)

    # 7) Save a modeling table and predictions
    base.to_csv(os.path.join(OUT_DIR, "modeling_table.csv"), index=False)

    pred_frame = test[["Player", "Team", "Position", "Age", true_fee]].copy()
    for name, p in preds_out.items():
        pred_frame[f"pred_{name}_marketValue"] = p
    pred_frame.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    # 8) Feature importances for tree models (when available)
    fi_frames = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"feature": X_cols, "importance": model.feature_importances_})
            fi["model"] = name
            fi_frames.append(fi)
    if fi_frames:
        pd.concat(fi_frames).sort_values(["model", "importance"], ascending=[True, False]).to_csv(
            os.path.join(OUT_DIR, "feature_importances.csv"), index=False
        )

    print("=== RMSE vs actual fee (lower is better) ===")
    print(res_df.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
