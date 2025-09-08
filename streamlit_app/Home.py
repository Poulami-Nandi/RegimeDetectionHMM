# streamlit_app/Home.py
import io
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.regime_detection import detect_regimes

# ---------------------- Page config ----------------------
st.set_page_config(
    page_title="TSLA Regime Detection — HMM + Rules",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- Helpers --------------------------
@st.cache_data(show_spinner=False)
def fetch_full_history_tsla():
    df = yf.download("TSLA", period="max", auto_adjust=True, progress=False)
    df = df.rename_axis("Date").sort_index()
    return df[["Close"]].dropna()

def add_emas(df: pd.DataFrame, fast=20, slow=100) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=fast, adjust=False).mean()
    out["EMA100"] = out["Close"].ewm(span=slow, adjust=False).mean()
    return out

def subset_last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()

def plot_close_ema(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="TSLA Close",
                             mode="lines", line=dict(width=2, color="#111111")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20 (fast)",
                             mode="lines", line=dict(width=1.6, color="#de6f6f")))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA100"], name="EMA100 (slow)",
                             mode="lines", line=dict(width=1.6, color="#63b3a4")))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig

def plot_regimes_last(df_reg: pd.DataFrame, years: int, title: str) -> go.Figure:
    # last N years view with EMAs and shaded bears
    df_last = subset_last_years(df_reg[["Close"]], years)
    df_last = add_emas(df_last)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last["Close"], name="TSLA Close",
                             mode="lines", line=dict(width=2, color="#111111")))
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last["EMA20"], name="EMA20 (fast)",
                             mode="lines", line=dict(width=1.6, color="#de6f6f")))
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last["EMA100"], name="EMA100 (slow)",
                             mode="lines", line=dict(width=1.6, color="#63b3a4")))

    # Shade candidate (light) and confirmed (dark)
    def add_spans(mask: pd.Series, opacity: float):
        mask = mask.reindex(df_last.index).fillna(False).astype(bool)
        in_blk, start = False, None
        for i, (d, v) in enumerate(mask.items()):
            if v and not in_blk:
                in_blk, start = True, d
            if in_blk and (not v or i == len(mask) - 1):
                end = d if not v else d
                fig.add_vrect(x0=start, x1=end, fillcolor="#d62728",
                              opacity=opacity, layer="below", line_width=0)
                in_blk = False

    if "bear_candidate" in df_reg.columns:
        add_spans(df_reg["bear_candidate"], 0.12)
    if "bear_confirm" in df_reg.columns:
        add_spans(df_reg["bear_confirm"], 0.30)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig

def download_png_button(fig: go.Figure, label: str, filename: str):
    try:
        png = fig.to_image(format="png", scale=6, width=2000, height=900)
        st.download_button(label, data=png, file_name=filename, mime="image/png")
    except Exception:
        with st.expander("PNG download (host is missing kaleido)"):
            st.info("PNG export needs `kaleido`. If not present, use the camera icon on the chart toolbar.")

# ---------------------- Sidebar (KEEP YOUR KNOBS) --------
st.sidebar.header("Controls")
zoom_years = st.sidebar.slider("Zoom window (years)", 1, 10, 3, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Regime knobs")
# Keep the same defaults you had earlier
n_components = st.sidebar.selectbox("HMM states (n_components)", [2, 3, 4], index=2)
ema_span = st.sidebar.slider("EMA smoothing of bear prob (ema_span)", 5, 50, 20)
bear_enter = st.sidebar.slider("Bear enter threshold", 0.50, 0.99, 0.80, 0.01)
bear_exit  = st.sidebar.slider("Bear exit threshold", 0.10, 0.90, 0.55, 0.01)
mom_thr    = st.sidebar.slider("Trend weakness (mom_threshold)", 0.00, 0.10, 0.03, 0.005)
dd_thr     = st.sidebar.slider("Drawdown confirm (ddown_threshold)", 0.00, 0.40, 0.15, 0.01)
confirm_d  = st.sidebar.slider("Confirm days", 1, 15, 7, 1)
bull_mom   = st.sidebar.slider("Bull trend (bull_mom_threshold)", 0.00, 0.05, 0.01, 0.005)
bull_dd_ex = st.sidebar.slider("Bull dd exit (bull_ddown_exit)", 0.00, 0.20, 0.06, 0.01)
confirm_db = st.sidebar.slider("Confirm days (bull)", 1, 10, 3, 1)
min_bear_run = st.sidebar.slider("Min bear run (days)", 1, 30, 15, 1)
min_bull_run = st.sidebar.slider("Min bull run (days)", 1, 30, 5, 1)
direction_gate = st.sidebar.checkbox("Use direction gate", value=True)
entry_lb = st.sidebar.number_input("Entry return lookback (days)", 1, 60, 10)
entry_ret = st.sidebar.number_input("Entry return thresh (≤ negative)", -0.20, 0.00, -0.01, step=0.005, format="%.3f")
entry_dd  = st.sidebar.number_input("Entry drawdown thresh (≤ negative)", -0.50, 0.00, -0.03, step=0.005, format="%.3f")
trend_gate = st.sidebar.checkbox("Require price < EMA100 at entry", value=True)
trend_exit_cross = st.sidebar.checkbox("Exit bear when price crosses above EMA100", value=True)
bear_profit_exit = st.sidebar.number_input("Bear profit exit (bounce % from entry)", 0.00, 0.30, 0.05, step=0.005, format="%.3f")

# ---------------------- Data + header --------------------
full_px = fetch_full_history_tsla()
full_px = add_emas(full_px)
zoom_px = subset_last_years(full_px, zoom_years)

full_years = (full_px.index.max() - full_px.index.min()).days / 365.25

st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")
st.caption(
    f"Full-history price with EMAs (IPO→today) and last-{zoom_years}-years regime view. "
    f"Data via Yahoo Finance at runtime.\n\n"
    f"Diagnostic — full-history span: {full_years:,.1f}y; zoom window: {zoom_years}y"
)

# ---------------------- Chart 1: IPO → today --------------
fig_full = plot_close_ema(full_px, "TSLA — Close with EMA20 / EMA100 (IPO → today)")
st.plotly_chart(fig_full, use_container_width=True, theme="streamlit")
download_png_button(fig_full, "Download full-history chart (high-res)", "tsla_full_history.png")

# ---------------------- Chart 2: last N years --------------
fig_zoom = plot_close_ema(zoom_px, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)")
st.plotly_chart(fig_zoom, use_container_width=True, theme="streamlit")
download_png_button(fig_zoom, "Download zoom chart (high-res)", "tsla_last_years.png")

# ---------------------- Regimes: last N years only --------
with st.spinner("Running regime detection…"):
    df_reg, _ = detect_regimes(
        ticker="TSLA",                 # 🔒 fixed ticker (no user change)
        n_components=n_components,
        ema_span=ema_span,
        bear_enter=bear_enter,
        bear_exit=bear_exit,
        mom_threshold=mom_thr,
        ddown_threshold=dd_thr,
        confirm_days=confirm_d,
        bull_mom_threshold=bull_mom,
        bull_ddown_exit=bull_dd_ex,
        confirm_days_bull=confirm_db,
        min_bear_run=min_bear_run,
        min_bull_run=min_bull_run,
        direction_gate=direction_gate,
        entry_ret_lookback=entry_lb,
        entry_ret_thresh=entry_ret,
        entry_dd_thresh=entry_dd,
        trend_gate=trend_gate,
        trend_exit_cross=trend_exit_cross,
        bear_profit_exit=bear_profit_exit,
        start="2000-01-01",           # IPO-ish start
        end="today",                  # explicit end to avoid None-labels
    )

fig_reg = plot_regimes_last(df_reg, zoom_years,
                            f"TSLA — Regimes (last {zoom_years} years; light = candidate bear, dark = confirmed bear)")
st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")
download_png_button(fig_reg, "Download regimes chart (high-res)", "tsla_regimes_last_years.png")

# --------------------------------------------------------------------------------------
# Parameter explainer (plain English)
# --------------------------------------------------------------------------------------
def render_parameter_explainer():
    st.markdown("---")
    st.markdown("## Parameter explainer")
    st.markdown(
        "**How to read this:** each knob says *what it does*, *when to nudge it up/down*, "
        "and a **TSLA-flavored example** so it’s easy to picture."
    )
    st.markdown("### General")
    st.markdown(
        """
- **`n_components` (HMM states)** — how many hidden “modes” (calm-bull, high-vol-bull, crash-bear, chop).  
  Raise if one state mixes very different behavior; lower if states look redundant.  
  **TSLA tip:** 3–4 often separates “bull-high-vol” from “true drawdown bear”.

- **`ema_span`** — smoothing of the bear-probability line. Higher = steadier, lower = more sensitive.  
  **TSLA tip:** 20 keeps most short noise out while catching turns in weeks.

- **`k_forward`** — look-ahead *only* used to name states during training.  
  Bigger = slower, trendier labeling; smaller = faster, noisier.  
  **TSLA tip:** 7–10 works; 20 can be too slow for sharp 2022/2024 moves.
        """
    )
    st.markdown("### Hysteresis thresholds (candidate entries/exits)")
    st.markdown(
        """
- **`bear_enter`** — start a **bear candidate** when smoothed bear-prob ≥ this.  
  Higher = fewer, stronger candidates. **TSLA:** 0.80.

- **`bear_exit`** — leave bear when smoothed bear-prob ≤ this. Keep well below enter.  
  **TSLA:** 0.55 (pair with 0.80 enter).

- **Auto thresholds (`auto_thresholds`, `bear_target`, `auto_window_years`, `min_gap`)** — pick enter/exit so last N years have ≈ `bear_target` share in bear. Good when personality shifts (e.g., NVDA 2023+).
        """
    )
    st.markdown("### Bear confirmations (candidate → confirmed)")
    st.markdown(
        """
- **`mom_threshold`** — EMA20 must sit **below** EMA100 by this fraction (trend weakness).  
  **TSLA:** 0.03 (~3%).

- **`ddown_threshold`** — price must be at least this far below a recent peak (drawdown).  
  **TSLA:** 0.15.

- **`confirm_days`** — how long weakness must persist (safer but later). **TSLA:** 7.

- **`min_bear_run`** — drop tiny bear islands shorter than this (cleanup). **TSLA:** 15 days.
        """
    )
    st.markdown("### Bull confirmations & early exits")
    st.markdown(
        """
- **`bull_mom_threshold`** — EMA20 **above** EMA100 by this fraction to back a bull pocket. **TSLA:** 0.01.  
- **`bull_ddown_exit`** — if drawdown has healed to within this of the peak, exit bear (even if prob lags). **TSLA:** 0.06.  
- **`confirm_days_bull`** — persistence for bull pockets. **TSLA:** 2–3.  
- **`min_bull_run`** — drop micro bull blips.  
- **`bear_profit_exit`** — bounce +X% from **bear entry** price → exit early. **TSLA:** 0.05.
        """
    )
    st.markdown("### Gates (filters at bear entry/exit)")
    st.markdown(
        """
- **`direction_gate`** — require weak recent returns and a real drawdown to *start* a bear.  
  **TSLA:** last 10-day return ≤ −1% and drawdown ≤ −3%.

- **`trend_gate`** — only enter bear if price is **below** EMA100.

- **`trend_exit_cross`** — if price crosses **above** EMA100 while in bear, exit.
        """
    )
    st.markdown("### Quick tuning cheatsheet")
    st.markdown(
        """
- Too many false bears? Raise `bear_enter`, `mom_threshold`, `ddown_threshold`, `confirm_days`; enable `direction_gate`.  
- Bear exits too late? Lower `bear_exit` a touch, or raise `bull_ddown_exit`, enable `trend_exit_cross`, set `bear_profit_exit`.  
- Missing small bull pockets? Lower `bull_mom_threshold`/`confirm_days_bull`; reduce `min_bull_run`.  
- New personality? Turn on `auto_thresholds` with 3–5y window and a sensible `bear_target` (25–35%).
        """
    )

render_parameter_explainer()
st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
