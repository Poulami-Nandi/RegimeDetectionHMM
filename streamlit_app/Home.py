# app/streamlit_app.py
from __future__ import annotations

import sys
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import streamlit as st

# --- make local package importable ---
HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE, HERE.parent]  # app/ or project root
for c in CANDIDATES:
    if (c / "src").exists():
        sys.path.insert(0, str(c))
        break

from src.data_loader import get_price_data
from src.regime_detection import detect_regimes, BULL, BEAR

# ---------- plotting helpers ----------

BEAR_CONF_COLOR = "tab:red"     # confirmed bear (darker red)
BEAR_CAND_COLOR = "#f4a6a6"     # candidate bear (lighter red)
EMA20_COLOR = "tab:orange"
EMA100_COLOR = "tab:purple"

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _plot_price_with_ema(ax, price: pd.DataFrame, title: str):
    close = price["Close"]
    ema20 = _ema(close, 20)
    ema100 = _ema(close, 100)

    line_close, = ax.plot(price.index, close, lw=1.6, label="Close")
    ax.plot(price.index, ema20, lw=1.2, alpha=0.9, color=EMA20_COLOR, label="EMA 20")
    ax.plot(price.index, ema100, lw=1.2, alpha=0.9, color=EMA100_COLOR, label="EMA 100")

    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)

def _plot_regimes(ax, df: pd.DataFrame, bear_enter_used: float, title: str):
    # Price + EMAs for context
    price = pd.DataFrame({"Close": df["Close"]}, index=df.index)
    ema20 = _ema(price["Close"], 20)
    ema100 = _ema(price["Close"], 100)

    line_close, = ax.plot(df.index, df["Close"], lw=1.6, label="Close")
    ax.plot(df.index, ema20, lw=1.2, alpha=0.9, color=EMA20_COLOR, label="EMA 20")
    ax.plot(df.index, ema100, lw=1.2, alpha=0.9, color=EMA100_COLOR, label="EMA 100")

    ymin, ymax = df["Close"].min(), df["Close"].max()
    cand_mask = (df["p_bear_ema"].values >= bear_enter_used)
    ax.fill_between(df.index, ymin, ymax, where=cand_mask, step="pre",
                    alpha=0.12, color=BEAR_CAND_COLOR, label="Bear candidate (light)")
    bear_mask = (df["regime"].values == BEAR)
    ax.fill_between(df.index, ymin, ymax, where=bear_mask, step="pre",
                    alpha=0.25, color=BEAR_CONF_COLOR, label="Bear confirmed (dark)")

    ax.set_ylabel("Price")
    ax.set_title(title)

    legend_handles = [
        Line2D([0],[0], color=line_close.get_color(), lw=2, label="Close"),
        Line2D([0],[0], color=EMA20_COLOR, lw=2, label="EMA 20"),
        Line2D([0],[0], color=EMA100_COLOR, lw=2, label="EMA 100"),
        Patch(facecolor=BEAR_CAND_COLOR, alpha=0.8, label="Bear candidate (light)"),
        Patch(facecolor=BEAR_CONF_COLOR,  alpha=0.8, label="Bear confirmed (dark)"),
        Patch(facecolor="none", edgecolor="none", label="Bull (unshaded)"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)

def _counts_last_years(df: pd.DataFrame, used_enter: float, used_exit: float, years: int = 3):
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    dfz = df.loc[df.index >= start]
    if dfz.empty:
        return {"days": 0, "bear_cand": 0, "bear_conf": 0, "bull_cand": 0, "bull_conf": 0}
    total = len(dfz)
    bear_cand = int((dfz["p_bear_ema"] >= used_enter).sum())
    bull_cand = int((dfz["p_bear_ema"] <= used_exit).sum())
    bear_conf = int((dfz["regime"] == BEAR).sum())
    bull_conf = int((dfz["regime"] == BULL).sum())
    return {"days": total, "bear_cand": bear_cand, "bear_conf": bear_conf,
            "bull_cand": bull_cand, "bull_conf": bull_conf}

# ---------- Streamlit UI ----------

st.set_page_config(page_title="TSLA Regime Detection (HMM)", layout="wide")

st.title("Tesla (TSLA) — Regime Detection with HMM")
st.caption("Data via Yahoo Finance. Model: Gaussian HMM with robust labeling, hysteresis, confirmations, and direction/trend gates.")

with st.sidebar:
    st.header("Controls (TSLA only)")

    # Core HMM + smoothing
    n_components = st.slider("HMM components", 2, 6, 4, 1)
    ema_span = st.slider("EMA span (probability smoothing)", 5, 40, 20, 1)
    k_forward = st.slider("Label horizon k (days)", 3, 20, 10, 1)

    # Fixed gates for TSLA (defaults from your tuned setup)
    st.subheader("Hysteresis thresholds")
    bear_enter = st.slider("Bear ENTER (p_bear_ema ≥)", 0.50, 0.95, 0.80, 0.01)
    bear_exit = st.slider("Bear EXIT (p_bear_ema ≤)", 0.30, bear_enter - 0.01, 0.55, 0.01)

    st.subheader("Confirmations & run-lengths")
    mom_threshold = st.number_input("Bear trend confirm (mom_20_100 ≤ -x)", value=0.03, step=0.005, format="%.3f")
    ddown_threshold = st.number_input("Bear drawdown confirm (≤ -x)", value=0.15, step=0.01, format="%.2f")
    confirm_days = st.slider("Confirm days (bear)", 1, 10, 7, 1)

    bull_mom_threshold = st.number_input("Bull trend confirm (mom_20_100 ≥ x)", value=0.01, step=0.005, format="%.3f")
    bull_ddown_exit = st.number_input("Bull drawdown exit (≥ -x)", value=0.06, step=0.01, format="%.2f")
    confirm_days_bull = st.slider("Confirm days (bull)", 1, 10, 3, 1)

    min_bear_run = st.slider("Min bear run (days)", 1, 30, 15, 1)
    min_bull_run = st.slider("Min bull run (days)", 1, 30, 5, 1)

    st.subheader("Direction & Trend Gates")
    direction_gate = st.checkbox("Use direction gate", True)
    entry_ret_lookback = st.slider("Entry lookback (days)", 5, 30, 10, 1)
    entry_ret_thresh = st.number_input("Entry trailing return ≤", value=-0.01, step=0.005, format="%.3f")
    entry_dd_thresh = st.number_input("Entry drawdown ≤", value=-0.03, step=0.005, format="%.3f")

    trend_gate = st.checkbox("Require trend gate at entry", True)
    req_close_below_100 = st.checkbox("Require Close < EMA100", True)
    req_ema20_below_100 = st.checkbox("Require EMA20 < EMA100", True)
    trend_exit_cross = st.checkbox("Exit bear on trend cross", True)

    bear_profit_exit = st.number_input("Bear bounce exit (since-entry ≥)", value=0.05, step=0.01, format="%.2f")

    st.subheader("Strict filter (presentation mode)")
    strict_direction = st.checkbox("Drop any bear that didn’t lose", False)
    strict_bear_min_ret = st.number_input("Strict: min total return ≤", value=-0.005, step=0.005, format="%.3f")
    strict_bear_min_maxdd = st.number_input("Strict: min worst drawdown ≤", value=-0.03, step=0.005, format="%.3f")

    run_btn = st.button("Run detection")

# Always pull fresh (IPO → today)
ticker = "TSLA"
today_str = dt.date.today().isoformat()
price = get_price_data(ticker, start=None, end=None)  # yfinance handles IPO start

# Chart 1 — Price + EMAs (full history)
st.subheader("1) Close price with EMA(20/100) — since IPO to today")
fig1, ax1 = plt.subplots(figsize=(12, 5))
_plot_price_with_ema(ax1, price, title=f"{ticker} — Close with EMA20/EMA100 (up to {today_str})")
st.pyplot(fig1, clear_figure=True)

# Run detection when page loads and when button pressed
if run_btn or True:
    df, model = detect_regimes(
        ticker=ticker,
        n_components=int(n_components),
        ema_span=int(ema_span),
        # fixed thresholds for TSLA
        bear_enter=float(bear_enter),
        bear_exit=float(bear_exit),
        # confirmations
        mom_threshold=float(mom_threshold),
        ddown_threshold=float(ddown_threshold),
        confirm_days=int(confirm_days),
        bull_mom_threshold=float(bull_mom_threshold),
        bull_ddown_exit=float(bull_ddown_exit),
        confirm_days_bull=int(confirm_days_bull),
        # run lengths
        min_bear_run=int(min_bear_run),
        min_bull_run=int(min_bull_run),
        # labeling horizon
        k_forward=int(k_forward),
        # direction & trend gates
        direction_gate=bool(direction_gate),
        entry_ret_lookback=int(entry_ret_lookback),
        entry_ret_thresh=float(entry_ret_thresh),
        entry_dd_thresh=float(entry_dd_thresh),
        trend_gate=bool(trend_gate),
        require_close_below_sma100=bool(req_close_below_100),
        require_sma20_below_sma100=bool(req_ema20_below_100),
        trend_exit_cross=bool(trend_exit_cross),
        bear_profit_exit=float(bear_profit_exit),
        # strict filter (optional)
        strict_direction=bool(strict_direction),
        strict_bear_min_ret=float(strict_bear_min_ret),
        strict_bear_min_maxdd=float(strict_bear_min_maxdd),
    )
    th = getattr(model, "thresholds_", {"bear_enter": bear_enter, "bear_exit": bear_exit})
    used_enter, used_exit = float(th["bear_enter"]), float(th["bear_exit"])

    # Chart 2 — Full-history regime plot
    st.subheader("2) Regime detection — since IPO to today")
    title_full = (f"{ticker} regimes (k={k_forward}, EMA={ema_span}, enter={used_enter:.2f}, exit={used_exit:.2f}, "
                  f"min_bear={min_bear_run}, min_bull={min_bull_run}, "
                  f"mom_thr={mom_threshold}, dd_thr={ddown_threshold}, "
                  f"bull_mom_thr={bull_mom_threshold}, bull_dd_exit={bull_ddown_exit}, "
                  f"confirm_bear={confirm_days}, confirm_bull={confirm_days_bull}, "
                  f"dir_gate={direction_gate}, lbk={entry_ret_lookback}, "
                  f"entry_ret_thr={entry_ret_thresh}, entry_dd_thr={entry_dd_thresh}, "
                  f"trend_gate={trend_gate}, profit_exit={bear_profit_exit}, "
                  f"strict={strict_direction})")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    _plot_regimes(ax2, df, bear_enter_used=used_enter, title=title_full)
    st.pyplot(fig2, clear_figure=True)

    # Chart 3 — Last 3 years regime plot
    st.subheader("3) Regime detection — last 3 years")
    end = df.index.max()
    start = end - pd.DateOffset(years=3)
    df3 = df.loc[df.index >= start]
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    title_3y = title_full + " — last 3y"
    _plot_regimes(ax3, df3, bear_enter_used=used_enter, title=title_3y)
    st.pyplot(fig3, clear_figure=True)

    # Quick counts for last 3y
    counts = _counts_last_years(df, used_enter, used_exit, years=3)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last 3y days", counts["days"])
    c2.metric("Bear (candidate)", counts["bear_cand"])
    c3.metric("Bear (confirmed)", counts["bear_conf"])
    c4.metric("Bull (candidate)", counts["bull_cand"])
    c5.metric("Bull (confirmed)", counts["bull_conf"])

st.caption("Note: This is a research demo. Not investment advice.")
