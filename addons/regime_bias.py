# addons/regime_bias.py
from __future__ import annotations
import numpy as np
import pandas as pd

def apply_regime_bias(
    base: pd.Series,
    close: pd.Series,
    regime_df: pd.DataFrame,
    *,
    vol_span: int = 20,
    bull_k: float = 0.50,
    bear_k_conf: float = 1.00,
    bear_k_cand: float = 0.40,
):
    """
    Build a regime-aware bias (NumPy array) aligned to `base.index`.
    Returns:
        final_forecast : pd.Series (same index as base)
        bias_series    : pd.Series (per-step multiplicative tilt)
    """

    # ---- 1) Align everything to the baseline index
    idx = base.index
    base  = pd.Series(base,  index=idx, dtype="float64")
    close = pd.Series(close, index=idx, dtype="float64")

    # Regime booleans (missing -> False), aligned to idx
    cols = []
    if "bear_candidate" in regime_df.columns:
        cols.append("bear_candidate")
    if "bear_confirm" in regime_df.columns:
        cols.append("bear_confirm")

    if not cols:  # nothing to do
        bias_series = pd.Series(0.0, index=idx, name="regime_bias")
        return (base * (1.0 + bias_series)).rename("final_forecast"), bias_series

    reg = (regime_df.reindex(idx)[cols]).fillna(False).astype(bool)
    cand = reg.get("bear_candidate", pd.Series(False, index=idx))
    conf = reg.get("bear_confirm",   pd.Series(False, index=idx))

    # Candidate-only = candidate AND NOT confirmed
    cand_only = cand & (~conf)

    # ---- 2) Volatility proxy: EMA of absolute returns (stable, shape-safe)
    ret = close.pct_change().fillna(0.0)
    vol = ret.abs().ewm(span=vol_span, adjust=False).mean().fillna(0.0)

    # ---- 3) Build bias as a NumPy array (prevents Pandas broadcasting pitfalls)
    bias = np.zeros(len(idx), dtype="float64")

    # Dark red (confirmed bear) gets the larger negative tilt
    mask_conf = conf.values
    if mask_conf.any():
        bias[mask_conf] -= bear_k_conf * vol.values[mask_conf]

    # Light red (candidate-only) gets a smaller negative tilt
    mask_cand = cand_only.values
    if mask_cand.any():
        bias[mask_cand] -= bear_k_cand * vol.values[mask_cand]

    # Optional: small positive tilt in neutral/bull pockets if returns are positive
    neutral = (~cand) & (~conf) & (ret > 0.0)
    mask_neu = neutral.values
    if mask_neu.any():
        bias[mask_neu] += bull_k * vol.values[mask_neu]

    # ---- 4) Return aligned Series
    bias_series = pd.Series(bias, index=idx, name="regime_bias")
    final_forecast = (base * (1.0 + bias_series)).rename("final_forecast")
    return final_forecast, bias_series

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
    Robust to DataFrame/Series inputs and keeps a clean Series output.
    """
    # --- normalize to 1-D Series ---
    if isinstance(close, pd.DataFrame):
        # take the first column if a frame was passed
        close = close.iloc[:, 0]
    close = pd.Series(close, index=close.index)  # ensure Series
    close.name = close.name or "Close"

    if horizon < 1:
        horizon = 1

    # drift from exponentially-weighted mean of returns
    ret = close.pct_change()
    drift = ret.ewm(span=20, adjust=False).mean()

    # compounded projection over the next N days (shifted so it's a next-step forecast)
    comp = (1.0 + drift).rolling(horizon).apply(np.prod, raw=True) - 1.0
    comp = comp.shift(1)

    out = close * (1.0 + comp)
    out.name = "base_forecast"  # <-- avoid DataFrame.rename path
    return out
