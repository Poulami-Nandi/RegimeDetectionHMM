import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Try to use your pipeline
try:
    from src.regime_detection import detect_regimes
except Exception as e:
    detect_regimes = None
    PIPELINE_IMPORT_ERROR = e
else:
    PIPELINE_IMPORT_ERROR = None


# ---------- Page config ----------
st.set_page_config(page_title="TSLA Regime Detection", layout="wide")

st.title("TSLA Regime Detection — HMM + Human-Readable Rules")
st.caption("Price, EMAs, regimes since IPO — plus a zoomed last-3-years view. All data via Yahoo Finance at runtime.")

# ---------- Sidebar controls ----------
st.sidebar.header("Configuration")

with st.sidebar.expander("General", expanded=True):
    n_components = st.number_input(
        "Number of HMM states (n_components)",
        min_value=2, max_value=6, value=4, step=1,
        help="How many market 'moods' the model may discover. "
             "More states can separate 'high-vol bull' vs 'crash'. For TSLA, 3–4 often works."
    )
    ema_span = st.slider(
        "Smoothing span for bear probability (ema_span)",
        min_value=5, max_value=60, value=20, step=1,
        help="Smoother signal -> fewer flips. This is an EMA over daily bear-probabilities."
    )
    k_forward = st.slider(
        "Horizon to judge state behavior (k_forward)",
        min_value=1, max_value=30, value=10, step=1,
        help="Used only to NAME states bear/bull by looking at near-future behavior, not for trading."
    )

with st.sidebar.expander("Hysteresis thresholds", expanded=True):
    bear_enter = st.slider(
        "Enter bear when smoothed prob ≥ (bear_enter)",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="Raise to see fewer candidate bears."
    )
    bear_exit = st.slider(
        "Exit bear when smoothed prob ≤ (bear_exit)",
        min_value=0.10, max_value=0.90, value=0.55, step=0.01,
        help="Higher exit exits earlier; bigger gap between enter/exit reduces flip-flops."
    )

with st.sidebar.expander("Bear confirmations (downside risk checks)", expanded=True):
    mom_threshold = st.slider(
        "Momentum threshold (fast under slow by ~X of price)",
        min_value=0.0, max_value=0.10, value=0.03, step=0.001,
        help="Approx. 3% means fast EMA sits ~3% under slow EMA."
    )
    ddown_threshold = st.slider(
        "Drawdown threshold from recent peak",
        min_value=0.0, max_value=0.50, value=0.15, step=0.01,
        help="Require price to be this far below the recent high to confirm bear."
    )
    confirm_days = st.slider(
        "Confirm days (bear)",
        min_value=1, max_value=20, value=7, step=1,
        help="Consecutive days the confirm(s) must hold before promoting candidate→confirmed."
    )

with st.sidebar.expander("Bull confirms & early exits", expanded=False):
    bull_mom_threshold = st.slider(
        "Bull momentum threshold (fast over slow by ~X of price)",
        min_value=0.0, max_value=0.10, value=0.01, step=0.001,
        help="Helps exit bear sooner when tide turns."
    )
    bull_ddown_exit = st.slider(
        "Bull drawdown exit (distance from peak shrinks to)",
        min_value=0.0, max_value=0.20, value=0.06, step=0.01,
        help="Exit bear if drawdown recovers toward the prior peak."
    )
    confirm_days_bull = st.slider(
        "Confirm days (bull)",
        min_value=1, max_value=10, value=3, step=1,
        help="Consecutive days bull confirms must hold to exit bear."
    )
    bear_profit_exit = st.slider(
        "Bounce/Profit exit from bear (+X from bear entry)",
        min_value=0.0, max_value=0.20, value=0.05, step=0.005,
        help="If TSLA rallies +X since bear entry, exit bear quickly even if probability lags."
    )

with st.sidebar.expander("Gates & run-cleanup", expanded=False):
    direction_gate = st.checkbox(
        "Require recent weakness at entry (direction_gate)",
        value=True,
        help="Only enter bear if last L days lost at least entry_ret_thresh AND drawdown ≤ entry_dd_thresh."
    )
    entry_ret_lookback = st.slider(
        "Lookback L for entry direction (days)",
        min_value=5, max_value=30, value=10, step=1,
        help="Used when direction_gate is on."
    )
    entry_ret_thresh = st.slider(
        "Cumulative return over L days must be ≤",
        min_value=-0.10, max_value=0.05, value=-0.01, step=0.001,
        help="Example −0.01 ≈ −1% over last L days."
    )
    entry_dd_thresh = st.slider(
        "Current drawdown must be ≤",
        min_value=-0.30, max_value=0.00, value=-0.03, step=0.005,
        help="Example −0.03 ≈ 3% below recent peak."
    )

    trend_gate = st.checkbox(
        "Require under-trend at bear entry (trend_gate)", value=False,
        help="Price or fast EMA must be under the slow EMA at entry."
    )
    trend_exit_cross = st.checkbox(
        "Exit bear on trend cross up", value=True,
        help="Exit bear when price/fast EMA crosses back above slow EMA."
    )

    min_bear_run = st.slider(
        "Min confirmed-bear run length (days)",
        min_value=1, max_value=40, value=15, step=1,
        help="Delete tiny islands shorter than this; cleaner plot."
    )
    min_bull_run = st.slider(
        "Min bull run length (days)",
        min_value=1, max_value=20, value=5, step=1,
        help="Delete tiny bull islands; optional cleanup."
    )

with st.sidebar.expander("Auto thresholds (optional, adaptive)", expanded=False):
    auto_thresholds = st.checkbox(
        "Auto-pick bear_enter/bear_exit from recent history", value=False,
        help="Good for AAPL/NVDA behavior shifts; TSLA can stay manual."
    )
    bear_target = st.slider(
        "Target bear share (recent window)",
        min_value=0.05, max_value=0.60, value=0.32, step=0.01,
        help="Used only if auto_thresholds is on."
    )
    auto_window_years = st.slider(
        "Auto window (years)",
        min_value=2, max_value=10, value=5, step=1,
        help="Recent window to learn thresholds from."
    )
    min_gap = st.slider(
        "Min gap between enter and exit",
        min_value=0.02, max_value=0.40, value=0.10, step=0.01,
        help="Prevents enter≈exit which causes flip-flops."
    )

with st.sidebar.expander("Figure settings", expanded=False):
    fig_w = st.slider("Figure width", 8, 20, 16, 1)
    fig_h = st.slider("Figure height", 4, 10, 7, 1)
    dpi = st.slider("DPI (clarity)", 100, 600, 300, 50)
    zoom_years = st.slider("Zoom window (last N years)", 1, 5, 3, 1)

st.sidebar.info(
    "ℹ️ **How to read the labels**\n\n"
    "- **Light red** = candidate bear (probability crossed ENTER).\n"
    "- **Dark red** = confirmed bear (weak trend and/or real drawdown persisted).\n"
    "- Exit bear quickly on bounce, trend cross, or when probability slips below EXIT."
)

# ---------- Guardrails ----------
if detect_regimes is None:
    st.error(
        "Couldn't import pipeline: `from src.regime_detection import detect_regimes` failed.\n\n"
        f"Error: {PIPELINE_IMPORT_ERROR}\n\n"
        "Make sure your repo layout matches the project (src/regime_detection.py) and try again."
    )
    st.stop()

# ---------- Run pipeline for TSLA ----------
# We always run TSLA per your spec (IPO ~2010-06-29)
ticker = "TSLA"

with st.spinner("Downloading data & computing regimes..."):
    df, model = detect_regimes(
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
        return_debug=True,           # ensure we get debug cols if your function supports it
        end="today",                 # explicit end label instead of None
    )

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Pipeline returned an empty DataFrame. Please try different settings.")
    st.stop()

# Ensure we have EMAs for the first plot
if "sma20" not in df.columns:
    df["sma20"] = df["Close"].ewm(span=20, adjust=False).mean()
if "sma100" not in df.columns:
    df["sma100"] = df["Close"].ewm(span=100, adjust=False).mean()

# ---------- Plot 1: Close + EMAs ----------
st.subheader("Close price with fast/slow EMAs (context)")
fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
ax1.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.2)
ax1.plot(df.index, df["sma20"], label="EMA20 (fast)", linewidth=1)
ax1.plot(df.index, df["sma100"], label="EMA100 (slow)", linewidth=1)
ax1.set_title(f"{ticker} — Close with EMA20 / EMA100")
ax1.set_xlabel("Date"); ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.25)
st.pyplot(fig1, use_container_width=True)

# ---------- Helpers for shading regimes ----------
def _shade_regions(ax, dates, mask, color, alpha, label):
    # Draw contiguous shaded spans for True regions in mask
    if mask is None or mask.sum() == 0:
        return
    in_run = False
    start = None
    for i, on in enumerate(mask.values):
        if on and not in_run:
            in_run = True
            start = dates[i]
        if in_run and (not on or i == len(mask) - 1):
            end = dates[i] if not on else dates[i]
            ax.axvspan(start, end, color=color, alpha=alpha, label=label if start == dates[mask.values.argmax()] else None)

# ---------- Plot 2: Full-history regimes ----------
st.subheader("Regimes — full history")
fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
ax2.plot(df.index, df["Close"], color="black", linewidth=1.0, label=f"{ticker} Close")
ax2.plot(df.index, df["sma20"], color="tab:blue", linewidth=0.8, alpha=0.7, label="EMA20")
ax2.plot(df.index, df["sma100"], color="tab:orange", linewidth=0.8, alpha=0.7, label="EMA100")

# Confirmed bear shading
confirmed_mask = None
if "regime" in df.columns:
    # convention: 1 = bear, 0 = bull
    confirmed_mask = df["regime"] == 1

# Candidate bear shading (lighter)
candidate_mask = None
if "p_bear_ema" in df.columns:
    candidate_mask = df["p_bear_ema"] >= bear_enter
    # If bear_confirm exists, only show candidate where NOT confirmed
    if "bear_confirm" in df.columns:
        candidate_mask = candidate_mask & (~df["bear_confirm"].astype(bool))
    if confirmed_mask is not None:
        candidate_mask = candidate_mask & (~confirmed_mask)

if candidate_mask is None and "bear_candidate" in df.columns:
    candidate_mask = df["bear_candidate"].astype(bool)

# Shade
if candidate_mask is not None and candidate_mask.any():
    _shade_regions(ax2, df.index, candidate_mask, color="red", alpha=0.25, label="Bear (candidate)")
if confirmed_mask is not None and confirmed_mask.any():
    _shade_regions(ax2, df.index, confirmed_mask, color="red", alpha=0.50, label="Bear (confirmed)")

ax2.set_title(f"{ticker} — Regimes (light red = candidate, dark red = confirmed)")
ax2.set_xlabel("Date"); ax2.set_ylabel("Price (USD)")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.25)
st.pyplot(fig2, use_container_width=True)

# ---------- Plot 3: Last N years zoom ----------
st.subheader(f"Regimes — last {zoom_years} years (same thresholds)")
cutoff = df.index.max() - pd.DateOffset(years=int(zoom_years))
dfz = df[df.index >= cutoff].copy()

fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
ax3.plot(dfz.index, dfz["Close"], color="black", linewidth=1.0, label=f"{ticker} Close")
ax3.plot(dfz.index, dfz["sma20"], color="tab:blue", linewidth=0.8, alpha=0.7, label="EMA20")
ax3.plot(dfz.index, dfz["sma100"], color="tab:orange", linewidth=0.8, alpha=0.7, label="EMA100")

conf_z = (dfz["regime"] == 1) if "regime" in dfz.columns else None
cand_z = None
if "p_bear_ema" in dfz.columns:
    cand_z = dfz["p_bear_ema"] >= bear_enter
    if "bear_confirm" in dfz.columns:
        cand_z = cand_z & (~dfz["bear_confirm"].astype(bool))
    if conf_z is not None:
        cand_z = cand_z & (~conf_z)
if cand_z is None and "bear_candidate" in dfz.columns:
    cand_z = dfz["bear_candidate"].astype(bool)

if cand_z is not None and cand_z.any():
    _shade_regions(ax3, dfz.index, cand_z, color="red", alpha=0.25, label="Bear (candidate)")
if conf_z is not None and conf_z.any():
    _shade_regions(ax3, dfz.index, conf_z, color="red", alpha=0.50, label="Bear (confirmed)")

ax3.set_title(f"{ticker} — Regimes (zoomed)")
ax3.set_xlabel("Date"); ax3.set_ylabel("Price (USD)")
ax3.legend(loc="upper left")
ax3.grid(alpha=0.25)
st.pyplot(fig3, use_container_width=True)

# ---------- Below-the-fold: detailed explainer ----------
st.markdown("---")
st.markdown("## Parameter explainer (plain English, TSLA examples)")
st.markdown(
    """
**Hysteresis (enter/exit):**  
- *Enter bear* when the **smoothed bear probability** goes above a high bar (e.g. **0.80**).  
- *Exit bear* when it falls below a lower bar (e.g. **0.55**).  
This spacing prevents constant flip-flopping on noise.

**Example (TSLA):** If bear probability is 78%→82%→85% while price grinds lower and sits under the EMA100, we create a **bear candidate**. If for ~**7 days** TSLA stays under trend and at least ~**15%** below its recent peak, we **confirm bear**. If price later rallies **+5%** from the bear entry or fast EMA crosses back above slow, we **exit** bear quickly.

**Confirmations & gates:**  
- **Momentum confirm:** fast EMA (e.g. 20-day) under slow (100-day) by ~**3%** for a few days.  
- **Drawdown confirm:** price ~**15%** below recent high for a few days.  
- **Direction gate:** at entry, last **10** days roughly **−1%** or worse **and** current drawdown about **−3%** or worse.  
- **Trend gate:** at entry, price/fast EMA below slow EMA.

**Run cleanup:**  
- **min_bear_run / min_bull_run:** delete tiny 1–2 day islands so the picture is readable.

**Bull confirmations (for exits):**  
- If trend flips up (fast>slow) and drawdown shrinks (e.g. within **6%** of the peak) for **3 days**, we treat it as bull confirmation and exit bear.

**Auto thresholds (optional):**  
Let the app *learn* realistic enter/exit from recent years (e.g. last 5y) with a target share of bear days (e.g. **32%**). Good for tickers like AAPL/NVDA when behavior shifts. TSLA often works fine with manual thresholds.

**What the colors mean:**  
- **Light red = candidate bear:** probability crossed ENTER but confirmations not complete.  
- **Dark red = confirmed bear:** weak trend/drawdown persisted for the required days.  
- **Unshaded = bull.**

**Interview one-liners you can use:**  
- “We separate signal discovery (HMM) from presentation rules (hysteresis, confirms, gates).”  
- “Bear = downside risk regime; if it rallies, we exit quickly.”  
- “Different tickers can have different knobs, but the story is consistent and explainable.”  
"""
)

# Footer
st.caption("Author: Dr. Poulami Nandi · Data: Yahoo Finance (yfinance) · This is a research demo, not investment advice.")
