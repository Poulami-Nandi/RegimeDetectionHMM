# streamlit_app/pages/2_Price_Prediction_with_Regimes.py
from __future__ import annotations
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import inspect

# Make repo root importable (no edits to existing code)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import your unchanged pipeline
from src.regime_detection import detect_regimes
from addons.regime_bias import base_forecast_close, apply_regime_bias

st.set_page_config(page_title="Regime-aware Forecast (TSLA demo)", layout="wide")
st.title("Regime-aware Price Forecast — TSLA (add-on demo)")
st.caption("This page is additive. It reuses the existing pipeline and does not modify the original app.")

# --------- Sidebar: locked to TSLA + familiar knobs ----------
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

# --------- Data fetch (direct full history for the forecast chart) ----------
@st.cache_data(ttl=3600, show_spinner=False)
def get_tsla():
    df = yf.download("TSLA", period="max", auto_adjust=True, progress=False)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Close"]].dropna()

px = get_tsla()
cutoff = px.index.max() - pd.DateOffset(years=zoom_years)
px_zoom = px.loc[px.index >= cutoff].copy()

# --------- Call your existing pipeline (unchanged) ----------
def _call_detect_regimes(func, **vals):
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

with st.spinner("Detecting regimes (unchanged pipeline)…"):
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

# Align to zoom window
reg_zoom = df_reg.reindex(px_zoom.index)

# --------- Build base forecast + regime bias ----------
base = base_forecast_close(px_zoom["Close"].squeeze(), horizon=5)
final, bias = apply_regime_bias(
    base=base, close=px_zoom["Close"], regime_df=reg_zoom,
    vol_span=vol_span, bull_k=bull_k, bear_k_conf=bear_k_conf, bear_k_cand=bear_k_cand,
)

# --------- Plot: regimes + forecasts ----------
def _segments(index, mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = index[i]
        if (not v) and (start is not None):
            out.append((start, index[i-1])); start=None
    if start is not None: out.append((start, index[-1]))
    return out

bear_cand = reg_zoom.get("bear_candidate", pd.Series(False, index=px_zoom.index)).astype(bool).values
bear_conf = reg_zoom.get("bear_confirm",   pd.Series(False, index=px_zoom.index)).astype(bool).values

ymin = float(px_zoom["Close"].min())*0.95
ymax = float(px_zoom["Close"].max())*1.05

fig = go.Figure()
fig.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["Close"], name="Close", line=dict(width=2, color="#111")))
fig.add_trace(go.Scatter(x=base.index,    y=base,             name="Base forecast", line=dict(width=1.8, dash="dot")))
fig.add_trace(go.Scatter(x=final.index,   y=final,            name="Regime-aware forecast", line=dict(width=2.2, color="#2a6f97")))
for s,e in _segments(px_zoom.index, bear_cand & (~bear_conf)):
    fig.add_vrect(x0=s, x1=e, y0=ymin, y1=ymax, fillcolor="crimson", opacity=0.12, line_width=0, layer="below")
for s,e in _segments(px_zoom.index, bear_conf):
    fig.add_vrect(x0=s, x1=e, y0=ymin, y1=ymax, fillcolor="crimson", opacity=0.30, line_width=0, layer="below")

fig.update_layout(
    title=f"TSLA — Regime-aware forecast (last {zoom_years}y; light=candidate, dark=confirmed)",
    template="plotly_white", height=650, margin=dict(l=10,r=10,t=60,b=10),
    legend=dict(orientation="h", x=0, y=1.03, bgcolor="rgba(255,255,255,0.85)"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", range=[ymin,ymax])
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("What’s happening here?", expanded=False):
    st.markdown(
        """
**Base forecast** is a lightweight EMA-drift projection (purely demonstrative).
We then add a **regime bias** that scales with recent volatility (σ):

- **Bull/sideways** → bias = `+ bull_k × σ`  (nudges forecast up)  
- **Bear (candidate)** → bias = `bear_k_cand × σ` (gentle down-bias)  
- **Bear (confirmed)** → bias = `bear_k_conf × σ` (stronger down-bias)

This page is read-only: it **does not change** the underlying detection code or Home page.
        """
    )
