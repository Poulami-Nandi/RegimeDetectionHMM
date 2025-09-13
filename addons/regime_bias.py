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


# addons/regime_bias.py  — replace ONLY this function

import numpy as np
import pandas as pd

def apply_regime_bias(
    base: pd.Series,
    close: pd.Series,
    regime_df: pd.DataFrame,
    *,
    vol_span: int = 20,
    bull_k: float = 0.60,
    bear_k_conf: float = 0.80,
    bear_k_cand: float = 0.30,
):
    """
    Combine a base forecast with a regime bias. All inputs are coerced to 1-D Series,
    indices are aligned, and boolean masks are reindexed (no broadcasting errors).

    Returns
    -------
    final : pd.Series  # same index as `base`
    bias  : pd.Series  # multiplicative bias applied, per timestamp (e.g., +0.60, -0.80, ...)
    """
    # --- normalize to Series ---
    if isinstance(base, pd.DataFrame):
        base = base.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    base = pd.Series(base, index=base.index).astype(float)
    close = pd.Series(close, index=close.index).astype(float)
    base.name = base.name or "base_forecast"
    close.name = close.name or "Close"

    # --- build a master index and align everything to it ---
    idx = base.index
    # Regime booleans (default False) aligned to base index
    cand = regime_df.get("bear_candidate", pd.Series(False, index=regime_df.index))
    conf = regime_df.get("bear_confirm",   pd.Series(False, index=regime_df.index))
    cand = pd.Series(cand, index=cand.index).astype(bool).reindex(idx, fill_value=False)
    conf = pd.Series(conf, index=conf.index).astype(bool).reindex(idx, fill_value=False)

    # bull = not in any bear state
    bull = (~cand) & (~conf)

    # --- optional: volatility if you want to scale later (kept here for future use) ---
    # vol = close.pct_change().ewm(span=vol_span, adjust=False).std().reindex(idx)
    # (currently unused in the bias calculation to keep it simple/deterministic)

    # --- construct a bias series aligned to base index ---
    bias = pd.Series(0.0, index=idx)

    # Assign piece-wise (scalar per mask) – no sequence-to-scalar errors
    bias.loc[bull] =  +float(bull_k)
    bias.loc[cand] =  -float(bear_k_cand)
    bias.loc[conf] =  -float(bear_k_conf)

    # --- produce final forecast ---
    final = base * (1.0 + bias)
    final.name = "final_forecast"
    bias.name = "regime_bias"

    return final, bias

