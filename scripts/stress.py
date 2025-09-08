import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

from src.regime_detection import detect_regimes
from src.utils import ensure_dir, get_logger
from src.features import make_features
from sklearn.preprocessing import StandardScaler

log = get_logger('stress')

def apply_shock(df: pd.DataFrame, shock: float = 0.0, vol_scale: float = 1.0) -> pd.DataFrame:
    """
    Create a shocked series by (1) scaling returns by 'vol_scale' and (2) adding a constant 'shock'.
    Then rebuild a shocked price path from the shocked returns.
    """
    out = df.copy()
    out['ret_shocked'] = df['ret'] * vol_scale + shock
    out['Close_shocked'] = df['Close'].iloc[0] * np.exp(out['ret_shocked'].cumsum())
    return out

def drift_metric(p: np.ndarray, q: np.ndarray) -> float:
    """
    Average Jensen–Shannon distance between two time series of regime-probability vectors.
    """
    n = min(len(p), len(q))
    if n == 0:
        return float('nan')
    d = [jensenshannon(p[i], q[i]) for i in range(n)]
    return float(np.nanmean(d))

def build_explained_rows(ticker: str, shock: float, vol_scale: float, js_drift_avg: float, alert: bool):
    return pd.DataFrame([{
        "ticker": ticker,
        "ticker_meaning": "Instrument symbol downloaded from Yahoo Finance (e.g., SPY, TSLA).",
        "shock": shock,
        "shock_meaning": "Additive daily return shock; -0.05 subtracts 5 percentage points from each day’s return.",
        "vol_scale": vol_scale,
        "vol_scale_meaning": "Volatility multiplier; 1.5 increases the size of daily moves by 50%.",
        "js_drift_avg": round(js_drift_avg, 6),
        "js_drift_avg_meaning": (
            "Average Jensen–Shannon distance (0..1) between baseline vs stressed regime probabilities. "
            "Higher = more structural change."
        ),
        "alert": alert,
        "alert_meaning": "True if drift exceeded the threshold (0.25 by default).",
        "method_note": (
            "We re-fit a fresh HMM on the shocked series then compare probability patterns vs baseline. "
            "For a fixed-model stress, score the shocked features with the baseline model instead."
        ),
    }])

def write_markdown_report(out_md: Path, row: pd.Series, threshold: float):
    md = f"""# Stress Test Report — {row['ticker']}

**Purpose:** Simulate a stressed market by modifying daily returns, then check how the model’s
**regime probabilities** differ from baseline.

## Inputs
- **Ticker:** `{row['ticker']}`
- **Shock:** `{row['shock']}` — Additive daily return shock (e.g., `-0.05` = -5% each day).
- **Volatility scale:** `{row['vol_scale']}` — Multiplier on typical day-to-day move size.

## Output
- **Average Jensen–Shannon drift (`js_drift_avg`):** `{row['js_drift_avg']}`
  - 0 = identical distributions; 1 = completely different.
- **Threshold:** `{threshold}`
- **Alert:** `{row['alert']}`

## Notes
- This script **re-fits** an HMM on the shocked series, then compares its regime probabilities to the baseline.  
- For **fixed-model** stress (no re-fit), score the shocked features using the baseline model instead; that usually yields larger drift under severe shocks.
"""
    out_md.write_text(md, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', default='SPY')
    ap.add_argument('--n_components', type=int, default=2)
    ap.add_argument('--shock', type=float, default=-0.03, help='Additive daily return shock')
    ap.add_argument('--vol_scale', type=float, default=1.2, help='Multiply return volatility')
    ap.add_argument('--outdir', default='reports')
    ap.add_argument('--alert_threshold', type=float, default=0.25, help='Threshold for JS drift alert')
    args = ap.parse_args()

    # Baseline (uses detect_regimes → returns model with scaler_ + feat_names_)
    base_df, base_model = detect_regimes(ticker=args.ticker, n_components=args.n_components)
    feat_names = getattr(base_model, 'feat_names_', ['ret','vol','ret_1','z_ret','mom_20_100','drawdown'])
    scaler = getattr(base_model, 'scaler_', None)

    # Shocked series
    shocked = apply_shock(base_df, shock=args.shock, vol_scale=args.vol_scale)
    shocked_feats = make_features(shocked[['Close_shocked']].rename(columns={'Close_shocked': 'Close'}))
    shocked_feats = shocked_feats.loc[base_df.index.intersection(shocked_feats.index)]
    X_shocked = shocked_feats[feat_names].values

    # Refit on shocked (with scaling)
    scaler_s = StandardScaler()
    Xs = scaler_s.fit_transform(X_shocked)
    from src.hmm_model import RegimeHMM
    model_s = RegimeHMM(n_components=args.n_components).fit(Xs)
    post_s = model_s.posterior(Xs)

    # Baseline posteriors (transform with baseline scaler if present)
    X_base = base_df[feat_names].values
    if scaler is not None:
        Xb = scaler.transform(X_base)
    else:
        Xb = X_base
    base_post = base_model.posterior(Xb)

    # Align and compare
    n = min(len(base_post), len(post_s))
    base_post = base_post[-n:]
    post_s = post_s[-n:]
    js = drift_metric(base_post, post_s)
    alert = bool(js > args.alert_threshold)

    ensure_dir(args.outdir)
    out_csv = Path(args.outdir) / f'stress_report_{args.ticker}.csv'
    out_csv_explained = Path(args.outdir) / f'stress_report_{args.ticker}_explained.csv'
    out_md = Path(args.outdir) / f'stress_report_{args.ticker}.md'

    base_row = pd.DataFrame([{
        'ticker': args.ticker,
        'shock': args.shock,
        'vol_scale': args.vol_scale,
        'js_drift_avg': round(js, 4),
        'alert': alert
    }])
    base_row.to_csv(out_csv, index=False)

    explained_df = build_explained_rows(args.ticker, args.shock, args.vol_scale, js, alert)
    explained_df.to_csv(out_csv_explained, index=False)
    write_markdown_report(out_md, explained_df.iloc[0], threshold=args.alert_threshold)

    print(base_row.to_string(index=False))
    print(f"\nSaved machine CSV  -> {out_csv}")
    print(f"Saved explained CSV -> {out_csv_explained}")
    print(f"Saved Markdown report -> {out_md}")

if __name__ == '__main__':
    main()
