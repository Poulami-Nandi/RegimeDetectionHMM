# streamlit_app/pages/2_Price_Prediction_with_Regimes.py
from __future__ import annotations

# ---- stdlib / path setup ----
import sys, os, inspect, importlib
from pathlib import Path

# ---- third-party ----
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# ========= repo import path (no edits to existing code) =========
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ========= unchanged pipeline =========
from src.regime_detection import detect_regimes

# ========= regime bias (reload to pick up latest shape-safe version) =========
import addons.regime_bias as _rb
importlib.reload(_rb)  # ensure we use the updated function definition
from addons.regime_bias import apply_regime_bias


# ========= Streamlit page =========
st.set_page_config(page_title="Regime-aware Forecast (TSLA demo)", layout="wide")
st.title("Regime-aware Price Forecast — TSLA (add-on demo)")
st.caption("This page is additive. It reuses the existing pipeline and does not modify the original app.")

# ---------------- Sidebar: locked to TSLA + familiar knobs ----------------
st.sidebar.header("Controls (TSLA demo)")
ticker = "TSLA"
zoom_years = st.sidebar.slider("Zoom window (years)", 1, 10, 3, 1)

st.sidebar.subheader("Regime knobs (same defaults)")
n_components        = st.sidebar.number_input("HMM states", 2, 6, 4, 1)
k_forward           = st.sidebar.slider("k_forward", 1, 20, 10, 1)
ema_span            = st.sidebar.slider("ema_span", 5, 60, 20, 1)
bear_enter          = st.sidebar.slider("bear_enter", 0.50, 0.99, 0.80, 0.01)
bear_exit           = st.sidebar.slider("bear_exit", 0.00, 0.95, 0.55, 0.01)
min_bear_run        = st.sidebar.slider("min_bear_run (days)", 1, 60, 15, 1)
min_bull_run        = st.sidebar.slider("min_bull_run (days)", 1, 60, 5, 1)
mom_threshold       = st.sidebar.slider("mom_threshold", 0.00, 0.10, 0.03, 0.001)
ddown_threshold     = st.sidebar.slider("ddown_threshold", 0.00, 0.30, 0.15, 0.005)
confirm_days        = st.sidebar.slider("confirm_days (bear)", 0, 20, 7, 1)
bull_mom_threshold  = st.sidebar.slider("bull_mom_threshold", 0.00, 0.05, 0.01, 0.001)
bull_ddown_exit     = st.sidebar.slider("bull_ddown_exit", 0.00, 0.20, 0.06, 0.005)
confirm_days_bull   = st.sidebar.slider("confirm_days (bull)", 0, 10, 3, 1)
direction_gate      = st.sidebar.checkbox("direction_gate", True)
trend_gate          = st.sidebar.checkbox("trend_gate", True)
strict              = st.sidebar.checkbox("strict", False)
entry_ret_lookback  = st.sidebar.slider("entry_ret_lookback", 1, 30, 10, 1)
entry_ret_thresh    = st.sidebar.slider("entry_ret_thresh", -0.05, 0.05, -0.01, 0.001)
entry_ddown_thresh  = st.sidebar.slider("entry_ddown_thresh", -0.10, 0.10, -0.03, 0.001)
bear_profit_exit    = st.sidebar.slider("bear_profit_exit", 0.00, 0.20, 0.05, 0.005)

st.sidebar.subheader("Bias layer")
vol_span     = st.sidebar.slider("vol_span (for bias σ)", 5, 60, 20, 1)
bull_k       = st.sidebar.slider("bull_k (×σ)", -1.0, 2.0, 0.60, 0.05)
bear_k_conf  = st.sidebar.slider("bear_k_conf (×σ)", -2.0, 0.0, -0.60, 0.05)
bear_k_cand  = st.sidebar.slider("bear_k_cand (×σ)", -2.0, 0.0, -0.30, 0.05)

with st.sidebar.expander("ARIMA baseline (advanced)"):
    p = st.number_input("ARIMA p", 0, 5, value=1, step=1)
    d = st.number_input("ARIMA d", 0, 2, value=1, step=1)
    q = st.number_input("ARIMA q", 0, 5, value=1, step=1)

# ========================= Helpers =========================
def _clean_series(s: pd.Series) -> pd.Series:
    """Coerce to float, drop NaNs, ensure 1-D series."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    return s.astype(float).dropna()

def _align_like(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reindex b to a's index; drop rows where either is NaN."""
    b2 = b.reindex(a.index)
    mask = ~(a.isna() | b2.isna())
    return a[mask], b2[mask]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv_last_years(ticker: str = "TSLA", years: int = 3) -> pd.DataFrame:
    """OHLCV for last N years (auto-adjusted)."""
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close_full(ticker: str = "TSLA") -> pd.DataFrame:
    """Full max history Close for slicing the zoom window index."""
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=True, progress=False)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Close"]].dropna()

def base_forecast_close_arima(ohlcv: pd.DataFrame, order=(1, 1, 1)) -> pd.Series:
    """
    One-step-ahead ARIMA baseline for the *historical* Close series.
    Returns a Series aligned to ohlcv.index named 'base_forecast'.
    """
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv:
        return pd.Series(dtype=float, name="base_forecast")

    close = ohlcv["Close"].astype(float)
    if len(close) < 50:
        return close.rename("base_forecast")

    try:
        model = ARIMA(close, order=order)
        res = model.fit()
        # one-step-ahead predicted mean; align to same index (fill first)
        pred = res.get_prediction(start=close.index[1], end=close.index[-1], dynamic=False)
        base = pred.predicted_mean.reindex(close.index)
        base.iloc[0] = close.iloc[0]
        return base.rename("base_forecast")
    except Exception as e:
        st.warning(f"ARIMA baseline failed ({e}). Falling back to a simple EMA(10).")
        return close.ewm(span=10, adjust=False).mean().rename("base_forecast")

def _call_detect_regimes(func, **vals):
    """Flexible mapper to whatever signature your detect_regimes exposes."""
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    out = {}
    def put(names, val):
        for n in names:
            if n in params:
                out[n] = val; return
    put(["ticker"],               vals["ticker"])
    put(["start"],                vals.get("start"))
    put(["end"],                  vals.get("end"))
    put(["n_components"],         vals["n_components"])
    put(["k_forward","k_fwd"],    vals["k_forward"])
    put(["ema_span","ema"],       vals["ema_span"])
    put(["bear_enter","enter_threshold","prob_threshold"], vals["bear_enter"])
    put(["bear_exit","exit_threshold","prob_exit"],        vals["bear_exit"])
    put(["min_bear_run","min_run"], vals["min_bear_run"])
    put(["min_bull_run"],          vals["min_bull_run"])
    put(["mom_threshold","mom_thr"], vals["mom_threshold"])
    put(["ddown_threshold","dd_thr"], vals["ddown_threshold"])
    put(["confirm_days","confirm_bear"], vals["confirm_days"])
    put(["bull_mom_threshold","bull_mom_thr"], vals["bull_mom_threshold"])
    put(["bull_ddown_exit","bull_dd_exit"], vals["bull_ddown_exit"])
    put(["confirm_days_bull","confirm_bull"], vals["confirm_days_bull"])
    put(["direction_gate"], vals["direction_gate"])
    put(["trend_gate"],     vals["trend_gate"])
    put(["entry_ret_lookback","lbk","lookback"], vals["entry_ret_lookback"])
    put(["entry_ret_thresh","entry_ret_thr"],    vals["entry_ret_thresh"])
    put(["entry_ddown_thresh","entry_dd_thr"],   vals["entry_ddown_thresh"])
    put(["bear_profit_exit","profit_exit"],      vals["bear_profit_exit"])
    put(["strict"],           vals["strict"])
    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)

# ========================= Data & pipeline =========================
with st.spinner("Fetching TSLA & detecting regimes…"):
    # Zoom index driven by full Close history
    px_full = fetch_close_full(ticker)
    cutoff  = px_full.index.max() - pd.DateOffset(years=zoom_years)
    px_zoom = px_full.loc[px_full.index >= cutoff].copy()

    # Unchanged pipeline over full history, reindexed to the zoom window
    df_reg, _ = _call_detect_regimes(
        detect_regimes,
        ticker=ticker, start="2000-01-01", end="today",
        n_components=n_components, k_forward=k_forward, ema_span=ema_span,
        bear_enter=bear_enter, bear_exit=bear_exit,
        min_bear_run=min_bear_run, min_bull_run=min_bull_run,
        mom_threshold=mom_threshold, ddown_threshold=ddown_threshold,
        confirm_days=confirm_days,
        bull_mom_threshold=bull_mom_threshold, bull_ddown_exit=bull_ddown_exit,
        confirm_days_bull=confirm_days_bull,
        direction_gate=direction_gate, trend_gate=trend_gate,
        entry_ret_lookback=entry_ret_lookback, entry_ret_thresh=entry_ret_thresh,
        entry_ddown_thresh=entry_ddown_thresh, bear_profit_exit=bear_profit_exit,
        strict=strict,
    )
    reg_zoom = df_reg.reindex(px_zoom.index)

# Normalize dtypes to strict booleans (avoids array/nested types)
for col in ["bear_candidate", "bear_confirm"]:
    if col in reg_zoom.columns:
        reg_zoom[col] = reg_zoom[col].astype(bool)

# ========================= ARIMA baseline on OHLCV (last N years) =========================
ohlcv = fetch_ohlcv_last_years(ticker, years=zoom_years)
if ohlcv.empty:
    st.error("Could not fetch OHLCV for ARIMA baseline. Try rerunning.")
    st.stop()

base     = base_forecast_close_arima(ohlcv, order=(p, d, q))     # Series
base     = pd.Series(base, index=base.index, dtype="float64")
px_zoom  = px_zoom.reindex(base.index)                           # align views to base
reg_zoom = reg_zoom.reindex(base.index)

# Final alignment cleanup
close = _clean_series(px_zoom["Close"])
close, base = _align_like(close, base)

# ========================= Regime-aware bias =========================
final, bias = apply_regime_bias(
    base=base,
    close=close,
    regime_df=reg_zoom,
    vol_span=vol_span,
    bull_k=bull_k,
    bear_k_conf=bear_k_conf,
    bear_k_cand=bear_k_cand,
)

# Align all three to the same index (and drop any residual NaNs)
close, base = _align_like(close, base)
close, final = _align_like(close, final)

if min(len(close), len(base), len(final)) < 5:
    st.warning("Not enough aligned points to plot after cleaning/alignment.")
    st.stop()

# ========================= Plot =========================
def _segments(index, mask_bool):
    out, start = [], None
    for i in range(len(index)):
        m = bool(mask_bool[i])
        if m and start is None:
            start = index[i]
        elif (not m) and (start is not None):
            out.append((start, index[i - 1])); start = None
    if start is not None:
        out.append((start, index[-1]))
    return out

bear_cand = reg_zoom.get("bear_candidate", pd.Series(False, index=close.index)).reindex(close.index).fillna(False).astype(bool).values
bear_conf = reg_zoom.get("bear_confirm",   pd.Series(False, index=close.index)).reindex(close.index).fillna(False).astype(bool).values
cand_only = bear_cand & (~bear_conf)

ymin = float(close.min()) * 0.95
ymax = float(close.max()) * 1.05

fig = go.Figure()
# background shading first (below lines)
for s, e in _segments(close.index, cand_only):
    fig.add_vrect(x0=s, x1=e, fillcolor="crimson", opacity=0.12, line_width=0, layer="below")
for s, e in _segments(close.index, bear_conf):
    fig.add_vrect(x0=s, x1=e, fillcolor="crimson", opacity=0.30, line_width=0, layer="below")

# lines
fig.add_trace(go.Scatter(
    x=close.index, y=close.values, name="Close",
    mode="lines", line=dict(width=1.9, color="#111")
))
fig.add_trace(go.Scatter(
    x=base.index, y=base.values, name=f"ARIMA({p},{d},{q}) base (one-step)",
    mode="lines", line=dict(width=2.0, dash="dash", color="#d62728")
))
fig.add_trace(go.Scatter(
    x=final.index, y=final.values, name="Regime-aware forecast",
    mode="lines", line=dict(width=2.4, color="#1f77b4")
))

fig.update_layout(
    title=f"TSLA — Regime-aware forecast (last {zoom_years}y; light=candidate, dark=confirmed)",
    template="plotly_white",
    height=540,
    margin=dict(l=10, r=10, t=80, b=10),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", range=[ymin, ymax]),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with st.expander("What’s happening here?", expanded=False):
    st.markdown(
        """
**Baseline:** ARIMA one-step-ahead projection on last-N-years daily **Close** (built from OHLCV).  
**Bias:** We tilt that baseline using regime signals and recent volatility (σ):

- **Bull/sideways** → bias ≈ `+ bull_k × σ` (gentle uplift)  
- **Bear (candidate)** → bias ≈ `bear_k_cand × σ` (small downward tilt)  
- **Bear (confirmed)** → bias ≈ `bear_k_conf × σ` (stronger downward tilt)

This page is read-only: it **does not change** the underlying detection code or the Home page.
        """
    )
