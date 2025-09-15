
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    fi   = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    preds= pd.read_csv(preds_path) if os.path.exists(preds_path) else None

    print("=== Files found ===")
    for p in [fc_path, inds_path, mdl_path, rmse_path, fi_path, preds_path]:
        print(("OK  " if os.path.exists(p) else "MISS"), p)

    #basic summaries/checks
    if fc is not None:
        print(f"\nFactor_Conversion shape: {fc.shape}  (players x 54 weeks expected)")
        row_stats = pd.DataFrame({
            'row_min': fc.min(axis=1),
            'row_max': fc.max(axis=1),
            'row_mean': fc.mean(axis=1)
        })
        print("Factor_Conversion row stats:\n", row_stats.describe().round(2))

    if inds is not None:
        print(f"\nPopularity_Indicators shape: {inds.shape}")
        print("PI columns:", inds.columns.tolist())

    if mdl is not None:
        print(f"\nModeling table shape: {mdl.shape}")
        pi_cols = ['PC1','Mean','Variance','Min','Max','Median']
        present = {c: (c in mdl.columns) for c in pi_cols}
        missing = {c: int(mdl[c].isna().sum()) if c in mdl.columns else None for c in pi_cols}
        print("PI present in modeling table:", present)
        print("PI missing counts:", missing)

        #Quick correlations (train: MV; test: Fee)
        train = mdl[mdl.get('Venta').isna()].copy()
        test  = mdl[mdl.get('Venta').notna()].copy()
        def s_corr(a, b):
            a = pd.to_numeric(a, errors='coerce')
            b = pd.to_numeric(b, errors='coerce')
            return float(a.corr(b))
        if len(train):
            print("\nCorrelations on TRAIN (vs MarketValue_eur):")
            for c in ['Min','Median','Mean','PC1']:
                if c in mdl.columns:
                    print(f"  {c:>6} : {s_corr(train[c], train['MarketValue_eur']): .3f}")
        if len(test):
            print("\nCorrelations on TEST (vs Cost_eur):")
            for c in ['Min','Median','Mean','PC1']:
                if c in mdl.columns:
                    print(f"  {c:>6} : {s_corr(test[c], test['Cost_eur']): .3f}")

    if rmse is not None:
        print("\n=== Model RMSE (lower is better) ===")
        print(rmse.sort_values('rmse_vs_fee'))

    #Simple visuals ---
    # 1) Histogram of Min
    if inds is not None and 'Min' in inds.columns:
        plt.figure()
        inds['Min'].plot(kind='hist', bins=40, title='Distribution of Popularity Indicator: Min')
        plt.xlabel('Min'); plt.ylabel('Count'); plt.tight_layout(); plt.show()

    #Predicted vs Actual (RF)
    if preds is not None and 'Cost_eur' in preds.columns and 'pred_RandomForest_marketValue' in preds.columns:
        plt.figure()
        x = pd.to_numeric(preds['Cost_eur'], errors='coerce')
        y = pd.to_numeric(preds['pred_RandomForest_marketValue'], errors='coerce')
        plt.scatter(x, y, alpha=0.6)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(lims, lims)
        plt.xlabel('Actual Transfer Fee (EUR)')
        plt.ylabel('Predicted Market Value (RF, EUR)')
        plt.title('RandomForest: Predicted vs Actual Fee')
        plt.tight_layout(); plt.show()

    #Predicted vs Actual (GBM)
    if preds is not None and 'Cost_eur' in preds.columns and 'pred_GBM_marketValue' in preds.columns:
        plt.figure()
        x = pd.to_numeric(preds['Cost_eur'], errors='coerce')
        y = pd.to_numeric(preds['pred_GBM_marketValue'], errors='coerce')
        plt.scatter(x, y, alpha=0.6)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(lims, lims)
        plt.xlabel('Actual Transfer Fee (EUR)')
        plt.ylabel('Predicted Market Value (GBM, EUR)')
        plt.title('GBM: Predicted vs Actual Fee')
        plt.tight_layout(); plt.show()

    #Top importances (if present)
    if fi is not None and {'feature','importance','model'}.issubset(set(fi.columns)):
        for m in sorted(fi['model'].unique()):
            top = fi[fi['model']==m].sort_values('importance', ascending=False).head(15)
            print(f"\nTop 15 {m} importances:")
            print(top[['feature','importance']])
            plt.figure()
            plt.barh(top['feature'][::-1], top['importance'][::-1])
            plt.xlabel('Importance'); plt.title(f'{m} Feature Importance (Top 15)')
            plt.tight_layout(); plt.show()

if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument('--outputs', default='/mnt/data', help='Folder containing the CSV outputs')
    # args = ap.parse_args()
    # main(args.outputs)
    default_dir = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default=default_dir, help='Folder containing the CSV outputs')
    args = ap.parse_args()
    main(args.outputs)
