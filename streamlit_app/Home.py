import os
import inspect
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Must be first for Streamlit
st.set_page_config(page_title="TSLA Regime Detection", layout="wide")

st.title("TSLA Regime Detection — HMM + Human-Readable Rules")
st.caption("Price, EMAs, regimes since IPO — plus a zoomed last-3-years view. Data via Yahoo Finance at runtime.")

# Try to import your pipeline
try:
    from src.regime_detection import detect_regimes
    PIPELINE_IMPORT_ERROR = None
except Exception as e:
    detect_regimes = None
    PIPELINE_IMPORT_ERROR = e

# ---------------- Sidebar controls ----------------
st.sidebar.header("Configuration")

with st.sidebar.expander("General", expanded=True):
    n_components = st.number_input(
        "Number of HMM states (n_components)", 2, 6, 4, 1,
        help="How many market 'moods' to learn. TSLA often works with 3–4."
    )
    ema_span = st.slider(
        "Smoothing span for bear probability (ema_span)", 5, 60, 20, 1,
        help="EMA over daily bear probabilities; bigger = smoother signal (fewer flips)."
    )
    k_forward = st.slider(
        "Horizon to judge state behavior (k_forward)", 1, 30, 10, 1,
        help="Used only to NAME states bear/bull from near-future behavior; not for trading."
    )

with st.sidebar.expander("Hysteresis thresholds", expanded=True):
    bear_enter = st.slider(
        "Enter bear when smoothed prob ≥ (bear_enter)", 0.50, 0.99, 0.80, 0.01,
        help="Raise to see fewer candidate bears."
    )
    bear_exit = st.slider(
        "Exit bear when smoothed prob ≤ (bear_exit)", 0.10, 0.90, 0.55, 0.01,
        help="Higher exit exits earlier; keep a gap vs enter to avoid flip-flops."
    )

with st.sidebar.expander("Bear confirmations (downside checks)", expanded=True):
    mom_threshold = st.slider(
        "Momentum threshold (fast under slow by ~X of price)", 0.0, 0.10, 0.03, 0.001,
        help="≈3% means EMA20 sits ~3% under EMA100 for some days."
    )
    ddown_threshold = st.slider(
        "Drawdown threshold from recent peak", 0.0, 0.50, 0.15, 0.01,
        help="Require price to be this far below the recent high to confirm bear."
    )
    confirm_days = st.slider(
        "Confirm days (bear)", 1, 20, 7, 1,
        help="Consecutive days momentum/drawdown must hold before promoting candidate→confirmed."
    )

with st.sidebar.expander("Bull confirms & early exits", expanded=False):
    bull_mom_threshold = st.slider(
        "Bull momentum threshold (fast over slow by ~X of price)", 0.0, 0.10, 0.01, 0.001,
        help="Helps exit bear when tide turns."
    )
    bull_ddown_exit = st.slider(
        "Bull drawdown exit (distance from peak shrinks to)", 0.0, 0.20, 0.06, 0.01,
        help="Exit bear if drawdown recovers toward the prior peak."
    )
    confirm_days_bull = st.slider(
        "Confirm days (bull)", 1, 10, 3, 1,
        help="Consecutive days bull confirms must hold to exit bear."
    )
    bear_profit_exit = st.slider(
        "Bounce/Profit exit from bear (+X from bear entry)", 0.0, 0.20, 0.05, 0.005,
        help="If TSLA rallies +X since bear entry, exit quickly even if probability lags."
    )

with st.sidebar.expander("Gates & run-cleanup", expanded=False):
    direction_gate = st.checkbox(
        "Require recent weakness at entry (direction_gate)", True,
        help="Enter bear only if last L days lost at least entry_ret_thresh AND drawdown ≤ entry_dd_thresh."
    )
    entry_ret_lookback = st.slider(
        "Lookback L for entry direction (days)", 5, 30, 10, 1,
        help="Used when direction_gate is on."
    )
    entry_ret_thresh = st.slider(
        "Cumulative return over L days must be ≤", -0.10, 0.05, -0.01, 0.001,
        help="Example −0.01 ≈ −1% over last L days."
    )
    entry_dd_thresh = st.slider(
        "Current drawdown must be ≤", -0.30, 0.00, -0.03, 0.005,
        help="Example −0.03 ≈ 3% below recent peak."
    )

    trend_gate = st.checkbox(
        "Require under-trend at bear entry (trend_gate)", False,
        help="Price or EMA20 must be under EMA100 at entry."
    )
    trend_exit_cross = st.checkbox(
        "Exit bear on trend cross up", True,
        help="Exit bear when price/EMA20 crosses back above EMA100."
    )

    min_bear_run = st.slider(
        "Min confirmed-bear run length (days)", 1, 40, 15, 1,
        help="Delete tiny islands shorter than this; cleaner plot."
    )
    min_bull_run = st.slider(
        "Min bull run length (days)", 1, 20, 5, 1,
        help="Delete tiny bull islands."
    )

with st.sidebar.expander("Auto thresholds (optional, adaptive)", expanded=False):
    auto_thresholds = st.checkbox(
        "Auto-pick bear_enter/bear_exit from recent history", False,
        help="Good for stocks whose behavior shifts (AAPL/NVDA); TSLA can stay manual."
    )
    bear_target = st.slider(
        "Target bear share (recent window)", 0.05, 0.60, 0.32, 0.01,
        help="Used only if auto_thresholds is on."
    )
    auto_window_years = st.slider(
        "Auto window (years)", 2, 10, 5, 1,
        help="Recent window to learn thresholds from."
    )
    min_gap = st.slider(
        "Min gap between enter and exit", 0.02, 0.40, 0.10, 0.01,
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
    "- Exit bear on bounce, trend cross, or when probability slips below EXIT."
)

# ---------------- Guardrails ----------------
if detect_regimes is None:
    st.error(
        "Couldn't import pipeline: `from src.regime_detection import detect_regimes` failed.\n\n"
        f"Error: {PIPELINE_IMPORT_ERROR}\n\n"
        "Ensure your repo has src/regime_detection.py with detect_regimes()."
    )
    st.stop()

# ---------------- Run pipeline for TSLA ----------------
ticker = "TSLA"

def call_detect_regimes_safely():
    """Pass only the kwargs your detect_regimes() supports.
       Also map common synonyms and fallback to a minimal call if needed.
    """
    sig = inspect.signature(detect_regimes)
    params = set(sig.parameters.keys())

    # Build the preferred kwargs
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
        return_debug=True,   # if supported
        end="today",         # if supported
    )

    # Add synonyms if the canonical names aren't present
    if "prob_threshold" in params and "bear_enter" not in params:
        pref["prob_threshold"] = float(bear_enter)
    if "prob_threshold_exit" in params and "bear_exit" not in params:
        pref["prob_threshold_exit"] = float(bear_exit)
    if "min_run" in params and "min_bear_run" not in params:
        pref["min_run"] = int(min_bear_run)
    if "ema_window" in params and "ema_span" not in params:
        pref["ema_window"] = int(ema_span)
    if "to" in params and "end" not in params:
        pref["to"] = "today"
    if "end_date" in params and "end" not in params:
        pref["end_date"] = "today"

    # Filter to only what the function accepts
    filtered = {k: v for k, v in pref.items() if k in params}

    try:
        return detect_regimes(**filtered)
    except TypeError as e:
        # As a last resort, try a minimal call that should work on older versions
        st.warning(
            "Using a minimal parameter set due to a version mismatch in detect_regimes().\n\n"
            f"Details: {e}"
        )
        return detect_regimes(ticker=ticker, n_components=int(n_components))

with st.spinner("Downloading data & computing regimes..."):
    df, model = call_detect_regimes_safely()

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Pipeline returned an empty DataFrame. Try different settings.")
    st.stop()

# Ensure EMAs exist for plotting (fallback if your pipeline doesn't add them)
if "sma20" not in df.columns:
    df["sma20"] = df["Close"].ewm(span=20, adjust=False).mean()
if "sma100" not in df.columns:
    df["sma100"] = df["Close"].ewm(span=100, adjust=False).mean()

# ---------------- Plot helpers ----------------
def shade_regions(ax, dates, mask, color, alpha, label):
    """Shade contiguous True regions in mask as spans."""
    if mask is None or not bool(mask.any()):
        return
    in_run = False
    start = None
    first = True
    for i, on in enumerate(mask.values):
        if on and not in_run:
            in_run = True
            start = dates[i]
        # Close the run if turning off or at last index
        is_last = (i == len(mask.values) - 1)
        if in_run and (not on or is_last):
            end = dates[i] if not on else dates[i]
            ax.axvspan(start, end, color=color, alpha=alpha, label=label if first else None)
            first = False
            in_run = False
            start = None

# ---------------- Plot 1: Close + EMAs ----------------
st.subheader("Close price with EMA20 / EMA100")
fig1, ax1 = plt.subplots(figsize=(int(fig_w), int(fig_h)), dpi=int(dpi))
ax1.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.2)
ax1.plot(df.index, df["sma20"], label="EMA20 (fast)", linewidth=1)
ax1.plot(df.index, df["sma100"], label="EMA100 (slow)", linewidth=1)
ax1.set_title(f"{ticker} — Close with EMA20 / EMA100")
ax1.set_xlabel("Date"); ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left"); ax1.grid(alpha=0.25)
st.pyplot(fig1, use_container_width=True)

# ---------------- Plot 2: Full-history regimes ----------------
st.subheader("Regimes — full history (light red = candidate, dark red = confirmed)")
fig2, ax2 = plt.subplots(figsize=(int(fig_w), int(fig_h)), dpi=int(dpi))
ax2.plot(df.index, df["Close"], color="black", linewidth=1.0, label=f"{ticker} Close")
ax2.plot(df.index, df["sma20"], color="tab:blue", linewidth=0.8, alpha=0.7, label="EMA20")
ax2.plot(df.index, df["sma100"], color="tab:orange", linewidth=0.8, alpha=0.7, label="EMA100")

confirmed_mask = (df["regime"] == 1) if "regime" in df.columns else None
candidate_mask = None
if "p_bear_ema" in df.columns:
    candidate_mask = df["p_bear_ema"] >= float(bear_enter)
    if "bear_confirm" in df.columns:
        candidate_mask = candidate_mask & (~df["bear_confirm"].astype(bool))
    if confirmed_mask is not None:
        candidate_mask = candidate_mask & (~confirmed_mask)
elif "bear_candidate" in df.columns:
    candidate_mask = df["bear_candidate"].astype(bool)

if candidate_mask is not None and candidate_mask.any():
    shade_regions(ax2, df.index, candidate_mask, color="red", alpha=0.25, label="Bear (candidate)")
if confirmed_mask is not None and confirmed_mask.any():
    shade_regions(ax2, df.index, confirmed_mask, color="red", alpha=0.50, label="Bear (confirmed)")

ax2.set_xlabel("Date"); ax2.set_ylabel("Price (USD)")
ax2.legend(loc="upper left"); ax2.grid(alpha=0.25)
st.pyplot(fig2, use_container_width=True)

# ---------------- Plot 3: Last N years zoom ----------------
st.subheader(f"Regimes — last {int(zoom_years)} years (same thresholds)")
cutoff = df.index.max() - pd.DateOffset(years=int(zoom_years))
dfz = df[df.index >= cutoff].copy()

fig3, ax3 = plt.subplots(figsize=(int(fig_w), int(fig_h)), dpi=int(dpi))
ax3.plot(dfz.index, dfz["Close"], color="black", linewidth=1.0, label=f"{ticker} Close")
ax3.plot(dfz.index, dfz["sma20"], color="tab:blue", linewidth=0.8, alpha=0.7, label="EMA20")
ax3.plot(dfz.index, dfz["sma100"], color="tab:orange", linewidth=0.8, alpha=0.7, label="EMA100")

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
    shade_regions(ax3, dfz.index, cand_z, color="red", alpha=0.25, label="Bear (candidate)")
if conf_z is not None and conf_z.any():
    shade_regions(ax3, dfz.index, conf_z, color="red", alpha=0.50, label="Bear (confirmed)")

ax3.set_xlabel("Date"); ax3.set_ylabel("Price (USD)")
ax3.legend(loc="upper left"); ax3.grid(alpha=0.25)
st.pyplot(fig3, use_container_width=True)

# ---------------- Explainer ----------------
st.markdown("---")
st.markdown("## Parameter explainer (plain English, with TSLA examples)")
st.markdown(
    """
**Hysteresis (enter/exit):** Enter bear when the **smoothed bear probability** goes above a high bar (e.g., 0.80). Exit when it falls below a lower bar (e.g., 0.55). This spacing prevents flip-flops.

**TSLA example:** If bear probability rises 78%→82%→85% while price grinds lower and sits under EMA100, we create a **bear candidate**. If for ~7 days TSLA stays under trend and ~15% below its recent peak, we **confirm bear**. If price later rallies +5% from entry or EMA20 crosses above EMA100, we **exit** bear quickly.

**Confirmations & gates:**  
- **Momentum confirm:** EMA20 under EMA100 by ~3% for several days.  
- **Drawdown confirm:** price ~15% below recent high for several days.  
- **Direction gate:** at entry, last 10 days ≲ −1% and drawdown ≲ −3%.  
- **Trend gate:** at entry, price/EMA20 below EMA100.

**Run cleanup:**  
- **min_bear_run / min_bull_run:** delete tiny 1–2 day islands.

**Bull confirmations (for exits):**  
- If trend flips up (EMA20 > EMA100) and drawdown shrinks (e.g., within 6% of peak) for 3 days, treat it as bull confirmation and exit bear.

**Auto thresholds (optional):** Learn enter/exit from recent years (e.g., last 5y) with a target share of bear days (e.g., 32%). Useful for changing behavior (AAPL/NVDA); TSLA often works fine manually.

**Color key:** Light red = candidate bear (probability crossed ENTER). Dark red = confirmed bear (weak trend/drawdown persisted). Unshaded = bull.
"""
)

st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
