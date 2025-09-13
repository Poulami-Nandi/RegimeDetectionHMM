# addons/regime_bias.py
from __future__ import annotations
import numpy as np
import pandas as pd

def regime_label(row: pd.Series) -> str:
    """Map your boolean flags to a simple label."""
    # prioritize confirmed bear, else candidate, else bull/sideways
    if bool(row.get("bear_confirm", False)):
        return "bear_confirm"
    if bool(row.get("bear_candidate", False)):
        return "bear_candidate"
    return "bull_or_sideways"

def rolling_vol(p: pd.Series, span: int = 20) -> pd.Series:
    r = p.pct_change()
    return r.ewm(span=span, adjust=False).std().fillna(method="bfill")

def base_forecast_close(close: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Very light, dependency-free baseline: EMA of returns → drift projection.
    This is only for demo; you can swap for ARIMA/XGBoost later.
    """
    ret = close.pct_change()
    drift = ret.ewm(span=20, adjust=False).mean()
    # project forward in simple compounded way
    f = (1 + drift).rolling(horizon).apply(np.prod, raw=True) - 1
    f = f.shift(1)  # next step forecast
    return (close * (1 + f)).rename("base_forecast")

def apply_regime_bias(
    base: pd.Series,
    close: pd.Series,
    regime_df: pd.DataFrame,
    vol_span: int = 20,
    bull_k: float = +0.60,
    bear_k_conf: float = -0.60,
    bear_k_cand: float = -0.30,
) -> pd.Series:
    """
    Combine base forecast with regime-aware bias proportional to recent vol.
    Bias is zero outside bear flags (treated as bull/sideways).
    """
    sigma = rolling_vol(close, span=vol_span).reindex(base.index).fillna(0.0)

    lbl = regime_df.reindex(base.index).apply(regime_label, axis=1)
    bias = pd.Series(0.0, index=base.index, name="bias")

    bias[lbl == "bear_confirm"]   = bear_k_conf * sigma[lbl == "bear_confirm"]
    bias[lbl == "bear_candidate"] = bear_k_cand * sigma[lbl == "bear_candidate"]
    bias[lbl == "bull_or_sideways"] = bull_k * sigma[lbl == "bull_or_sideways"]

    final = base * (1 + bias)
    final.name = "final_forecast"
    return final, bias.rename("regime_bias")
