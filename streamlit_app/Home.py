import os
import inspect
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
st.title("TSLA Regime Detection — HMM + Human-Readable Rules")
st.caption("Full-history price with EMAs (from IPO) + last-N-years regime view. Data via Yahoo Finance at runtime.")

# -------- Import your pipeline --------
try:
    from src.regime_detection import detect_regimes
    PIPELINE_IMPORT_ERROR = None
except Exception as e:
    detect_regimes = None
    PIPELINE_IMPORT_ERROR = e

# -------- Sidebar controls --------
st.sidebar.header("Configuration")

with st.sidebar.expander("General", expanded=True):
    n_components = st.number_input("HMM states (n_components)", 2, 6, 4, 1)
    ema_span = st.slider("Smoothing span for bear probability (ema_span)", 5, 60, 20, 1)
    k_forward = st.slider("Horizon to judge state behavior (k_forward)", 1, 30, 10, 1)

with st.sidebar.expander("Hysteresis thresholds", expanded=True):
    bear_enter = st.slider("Enter bear when smoothed prob ≥ (bear_enter)", 0.50, 0.99, 0.80, 0.01)
    bear_exit  = st.slider("Exit bear when smoothed prob ≤ (bear_exit)", 0.10, 0.90, 0.55, 0.01)

with st.sidebar.expander("Bear confirmations (downside checks)", expanded=True):
    mom_threshold = st.slider("Momentum threshold (EMA20 under EMA100 by ~X of price)", 0.0, 0.10, 0.03, 0.001)
    ddown_threshold = st.slider("Drawdown threshold from recent peak", 0.0, 0.50, 0.15, 0.01)
    confirm_days = st.slider("Confirm days (bear)", 1, 20, 7, 1)

with st.sidebar.expander("Bull confirms & early exits", expanded=False):
    bull_mom_threshold = st.slider("Bull momentum threshold (EMA20 over EMA100 by ~X)", 0.0, 0.10, 0.01, 0.001)
    bull_ddown_exit = st.slider("Bull drawdown exit (distance from peak shrinks to)", 0.0, 0.20, 0.06, 0.01)
    confirm_days_bull = st.slider("Confirm days (bull)", 1, 10, 3, 1)
    bear_profit_exit = st.slider("Bounce/Profit exit from bear (+X from bear entry)", 0.0, 0.20, 0.05, 0.005)

with st.sidebar.expander("Gates & run-cleanup", expanded=False):
    direction_gate = st.checkbox("Require recent weakness at entry (direction_gate)", True)
    entry_ret_lookback = st.slider("Lookback L for entry direction (days)", 5, 30, 10, 1)
    entry_ret_thresh = st.slider("Cumulative return over L days must be ≤", -0.10, 0.05, -0.01, 0.001)
    entry_dd_thresh = st.slider("Current drawdown must be ≤", -0.30, 0.00, -0.03, 0.005)
    trend_gate = st.checkbox("Require under-trend at bear entry (trend_gate)", False)
    trend_exit_cross = st.checkbox("Exit bear on trend cross up", True)
    min_bear_run = st.slider("Min confirmed-bear run length (days)", 1, 40, 15, 1)
    min_bull_run = st.slider("Min bull run length (days)", 1, 20, 5, 1)

with st.sidebar.expander("Auto thresholds (optional, adaptive)", expanded=False):
    auto_thresholds = st.checkbox("Auto-pick bear_enter/bear_exit from recent history", False)
    bear_target = st.slider("Target bear share (recent window)", 0.05, 0.60, 0.32, 0.01)
    auto_window_years = st.slider("Auto window (years)", 2, 10, 5, 1)
    min_gap = st.slider("Min gap between enter and exit", 0.02, 0.40, 0.10, 0.01)

with st.sidebar.expander("Figure settings", expanded=False):
    fig_w = st.slider("Figure width", 8, 20, 16, 1)
    fig_h = st.slider("Figure height", 4, 10, 7, 1)
    dpi = st.slider("DPI (clarity)", 100, 600, 300, 50)
    zoom_years = st.slider("Zoom window (last N years)", 1, 5, 3, 1)

st.sidebar.info(
    "ℹ️ Light red = candidate bear (probability crossed ENTER). Dark red = confirmed bear (weak trend/drawdown persisted)."
)

# -------- Guardrail on import failure --------
if detect_regimes is None:
    st.error(
        "Couldn't import pipeline: `from src.regime_detection import detect_regimes` failed.\n\n"
        f"Error: {PIPELINE_IMPORT_ERROR}\n"
        "Ensure src/regime_detection.py defines detect_regimes()."
    )
    st.stop()

ticker = "TSLA"

def call_detect_regimes_safely():
    sig = inspect.signature(detect_regimes)
    params = set(sig.parameters.keys())

    pref = dict(
        ticker=ticker,
        n_components=int(n_components),
        k_forward=int(k_forward),
        ema_span=int(ema_span),
        bear_enter=float(bear_enter),
        bear_exit=float(bear_exit),
        min_bear_run=int(min_bear_run),
        min_bull_run=int(min_bull_run),
        mom_threshold=float(mom_threshold),
        ddown_threshold=float(ddown_threshold),
        confirm_days=int(confirm_days),
        bull_mom_threshold=float(bull_mom_threshold),
        bull_ddown_exit=float(bull_ddown_exit),
        confirm_days_bull=int(confirm_days_bull),
        direction_gate=bool(direction_gate),
        entry_ret_lookback=int(entry_ret_lookback),
        entry_ret_thresh=float(entry_ret_thresh),
        entry_dd_thresh=float(entry_dd_thresh),
        trend_gate=bool(trend_gate),
        trend_exit_cross=bool(trend_exit_cross),
        auto_thresholds=bool(auto_thresholds),
        bear_target=float(bear_target),
        auto_window_years=int(auto_window_years),
        min_gap=float(min_gap),
        return_debug=True,     # if supported
        start="2010-01-01",    # ask for full history explicitly if supported
        end="today",           # explicit end label
    )

    # Synonyms for older versions
    if "prob_threshold" in params and "bear_enter" not in params:
        pref["prob_threshold"] = float(bear_enter)
    if "prob_threshold_exit" in params and "bear_exit" not in params:
        pref["prob_threshold_exit"] = float(bear_exit)
    if "min_run" in params and "min_bear_run" not in params:
        pref["min_run"] = int(min_bear_run)
    if "ema_window" in params and "ema_span" not in params:
        pref["ema_window"] = int(ema_span)
    if "from" in params and "start" not in params:
        pref["from"] = "2010-01-01"
    if "to" in params and "end" not in params:
        pref["to"] = "today"
    if "start_date" in params and "start" not in params:
        pref["start_date"] = "2010-01-01"
    if "end_date" in params and "end" not in params:
        pref["end_date"] = "today"
    if "full_history" in params:
        pref["full_history"] = True

    filtered = {k: v for k, v in pref.items() if k in params}

    try:
        return detect_regimes(**filtered)
    except TypeError as e:
        st.warning("Falling back to a minimal call due to a version mismatch.\n\n" + str(e))
        return detect_regimes(ticker=ticker, n_components=int(n_components))

with st.spinner("Downloading data & computing regimes..."):
    df, model = call_detect_regimes_safely()

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Pipeline returned an empty DataFrame. Try different settings.")
    st.stop()

# Ensure EMAs exist (fallbacks)
if "sma20" not in df.columns:
    df["sma20"] = df["Close"].ewm(span=20, adjust=False).mean()
if "sma100" not in df.columns:
    df["sma100"] = df["Close"].ewm(span=100, adjust=False).mean()

# ---------- Build a true full-history series for the FIRST chart ----------
# Some pipeline versions return only a recent window. If the span is too short,
# fetch full history via yfinance just for the first (IPO→today) chart.
def get_full_history_df():
    try:
        # If we already have ≥ 8 years, use it; otherwise fetch.
        if (df.index.max() - df.index.min()) >= pd.DateOffset(years=8):
            base = df.copy()
        else:
            import yfinance as yf
            raw = yf.download(ticker, start="2010-01-01", auto_adjust=True, progress=False)
            if raw.empty:
                base = df.copy()
            else:
                raw.rename(columns=str.title, inplace=True)  # Close, Open, etc.
                base = raw[["Close"]].copy()
        if "sma20" not in base.columns:
            base["sma20"] = base["Close"].ewm(span=20, adjust=False).mean()
        if "sma100" not in base.columns:
            base["sma100"] = base["Close"].ewm(span=100, adjust=False).mean()
        return base.dropna()
    except Exception:
        return df.copy()

df_full = get_full_history_df()

# ---------- Build last-N years slice for the other charts ----------
cutoff = df.index.max() - pd.DateOffset(years=int(zoom_years))
dfz = df[df.index >= cutoff].copy()

# ---------- Plot A: Full history Close + EMAs (IPO → today) ----------
st.subheader("TSLA — Close with EMA20 / EMA100 (IPO → today)")
fig_full, ax_full = plt.subplots(figsize=(int(fig_w), int(fig_h)), dpi=int(dpi))
ax_full.plot(df_full.index, df_full["Close"], label="TSLA Close", linewidth=1.2)
ax_full.plot(df_full.index, df_full["sma20"], label="EMA20 (fast)", linewidth=1)
ax_full.plot(df_full.index, df_full["sma100"], label="EMA100 (slow)", linewidth=1)
ax_full.set_xlabel("Date"); ax_full.set_ylabel("Price (USD)")
ax_full.legend(loc="upper left"); ax_full.grid(alpha=0.25)
st.pyplot(fig_full, use_container_width=True)

# ---------- Plot B: Close + EMAs (last N years only) ----------
st.subheader(f"TSLA — Close with EMA20 / EMA100 (last {int(zoom_years)} years)")
fig_last, ax_last = plt.subplots(figsize=(int(fig_w), int(fig_h)), dpi=int(dpi))
ax_last.plot(dfz.index, dfz["Close"], label="TSLA Close", linewidth=1.2)
ax_last.plot(dfz.index, dfz["sma20"], label="EMA20 (fast)", linewidth=1)
ax_last.plot(dfz.index, dfz["sma100"], label="EMA100 (slow)", linewidth=1)
ax_last.set_xlabel("Date"); ax_last.set_ylabel("Price (USD)")
ax_last.legend(loc="upper left"); ax_last.grid(alpha=0.25)
st.pyplot(fig_last, use_container_width=True)

# ---------- Plot C: Regimes (last N years only) ----------
def shade_regions(ax, dates, mask, color, alpha, label):
    if mask is None or not bool(mask.any()):
        return
    in_run, start, first = False, None, True
    vals = mask.values
    for i, on in enumerate(vals):
        if on and not in_run:
            in_run, start = True, dates[i]
        is_last = (i == len(vals) - 1)
        if in_run and (not on or is_last):
            end = dates[i]
            ax.axvspan(start, end, color=color, alpha=alpha, label=label if first else None)
            first, in_run, start = False, False, None

st.subheader(f"TSLA — Regimes (last {int(zoom_years)} years; light red=candidate, dark red=confirmed)")
fig_reg, ax_reg = plt.subplots(figsize=(int(fig_w), int(fig_h)), dpi=int(dpi))
ax_reg.plot(dfz.index, dfz["Close"], color="black", linewidth=1.0, label="TSLA Close")
ax_reg.plot(dfz.index, dfz["sma20"], color="tab:blue", linewidth=0.8, alpha=0.7, label="EMA20")
ax_reg.plot(dfz.index, dfz["sma100"], color="tab:orange", linewidth=0.8, alpha=0.7, label="EMA100")

conf_z = (dfz["regime"] == 1) if "regime" in dfz.columns else None
cand_z = None
if "p_bear_ema" in dfz.columns:
    cand_z = dfz["p_bear_ema"] >= float(bear_enter)
    if "bear_confirm" in dfz.columns:
        cand_z = cand_z & (~dfz["bear_confirm"].astype(bool))
    if conf_z is not None:
        cand_z = cand_z & (~conf_z)
elif "bear_candidate" in dfz.columns:
    cand_z = dfz["bear_candidate"].astype(bool)

if cand_z is not None and cand_z.any():
    shade_regions(ax_reg, dfz.index, cand_z, color="red", alpha=0.25, label="Bear (candidate)")
if conf_z is not None and conf_z.any():
    shade_regions(ax_reg, dfz.index, conf_z, color="red", alpha=0.50, label="Bear (confirmed)")

ax_reg.set_xlabel("Date"); ax_reg.set_ylabel("Price (USD)")
ax_reg.legend(loc="upper left"); ax_reg.grid(alpha=0.25)
st.pyplot(fig_reg, use_container_width=True)

# ---------- Explainer ----------
st.markdown("---")
st.markdown("## Parameter explainer")
st.markdown(
    """
**Acronyms**
- **HMM:** Hidden Markov Model (unsupervised model that assigns a hidden “state” to each day).
- **EMA:** Exponential Moving Average (recent prices weighted more than older prices).

**General**
- **n_components** — number of hidden states the HMM can use.  
  *TSLA example:* 4 states can separate “calm bull”, “fast-rising bull”, “choppy bear”, “crash”.
- **ema_span** — how much we smooth the daily bear probability. Bigger = steadier signal.  
  *TSLA example:* span=20 filters a lot of noise; span=8 reacts faster but flips more.
- **k_forward** — only for *naming* states (we look a few days ahead to decide if a state behaves like bull or bear).

**Hysteresis thresholds**
- **bear_enter** — create a **bear candidate** when smoothed bear probability rises above this (e.g., 0.80).  
- **bear_exit** — exit bear when the smoothed probability falls below this (e.g., 0.55).  
  Two different levels prevent flip-flops.

**Bear confirmations (turn candidate → confirmed)**
- **mom_threshold** — EMA20 under EMA100 by ~X of price (trend weakness).  
- **ddown_threshold** — price ~X below recent peak (drawdown).  
- **confirm_days** — how many consecutive days those must hold.

**Bull confirms & early exits (to end bear)**
- **bull_mom_threshold** — EMA20 above EMA100 by ~X.  
- **bull_ddown_exit** — drawdown recovered toward peak (e.g., within 6%).  
- **confirm_days_bull** — consecutive days those must hold.  
- **bear_profit_exit** — if price rallies +X% from bear entry, exit quickly even if probability lags.

**Gates & cleanup**
- **direction_gate** — only allow bear entry if last L days were weak (≤ entry_ret_thresh) **and** current drawdown ≤ entry_dd_thresh.  
- **trend_gate** — require price/EMA20 under EMA100 at entry.  
- **trend_exit_cross** — exit bear on cross up of price/EMA20 over EMA100.  
- **min_bear_run / min_bull_run** — delete tiny 1–2 day islands for readability.

**Auto thresholds (optional)**
- **auto_thresholds** — learn enter/exit from recent years with **bear_target** share.  
- **auto_window_years** — how far back to learn from.  
- **min_gap** — minimum spacing between enter and exit so they aren’t equal.

**Color key**
- Light red = candidate bear (probability crossed ENTER).  
- Dark red = confirmed bear (weak trend/drawdown persisted).  
"""
)
st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
