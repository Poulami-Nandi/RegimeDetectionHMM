# streamlit_app/Home.py
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import io
import sys
import importlib
import subprocess

from src.regime_detection import detect_regimes  # your pipeline

st.set_page_config(
    page_title="TSLA Regime Detection — HMM + Rules",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== Utils =====================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tsla_full_history() -> pd.DataFrame:
    """
    Robust full-history fetch for TSLA (IPO: 2010-06-29) used ONLY for the price/EMA charts.
    Avoids using the pipeline slice so full=IPO→today and zoom=last N years are truly different.
    """
    try:
        t = yf.Ticker("TSLA")
        df = t.history(period="max", interval="1d", auto_adjust=True)
        df = df.rename(columns={"Adj Close": "AdjClose"})
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[["Close"]].dropna()
        # Some hosts return a truncated series on first call; try a second fetch if tiny
        if len(df) < 100:
            df2 = yf.download("TSLA", period="max", auto_adjust=True, progress=False)
            if df2.index.tz is not None:
                df2.index = df2.index.tz_localize(None)
            df2 = df2[["Close"]].dropna()
            if len(df2) > len(df):
                df = df2
        return df
    except Exception as e:
        st.error(f"Yahoo fetch failed: {e}")
        return pd.DataFrame(columns=["Close"])

def add_emas(df: pd.DataFrame, fast=20, slow=100) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["EMA20"] = out["Close"].ewm(span=fast, adjust=False).mean()
    out["EMA100"] = out["Close"].ewm(span=slow, adjust=False).mean()
    return out

def last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()

def plot_close_emas(df: pd.DataFrame, title: str, h=440) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="TSLA Close",
                                 mode="lines", line=dict(width=2.2, color="#111")))
        if "EMA20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20 (fast)",
                                     mode="lines", line=dict(width=1.8, color="#de6f6f")))
        if "EMA100" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA100"], name="EMA100 (slow)",
                                     mode="lines", line=dict(width=1.8, color="#63b3a4")))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=10, t=46, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig

def add_bear_shading(fig: go.Figure, df_full: pd.DataFrame, df_view: pd.DataFrame,
                     col: str, opacity: float):
    # Shade intervals where df_full[col] is True but only across the df_view x-range
    if df_full.empty or df_view.empty or col not in df_full:
        return
    mask = df_full[col].reindex(df_view.index).fillna(False).astype(bool)
    in_blk, start = False, None
    idx = list(mask.index)
    for i, d in enumerate(idx):
        v = bool(mask.loc[d])
        last = (i == len(idx) - 1)
        if v and not in_blk:
            in_blk, start = True, d
        if in_blk and (not v or last):
            end = d
            fig.add_vrect(x0=start, x1=end, fillcolor="#d62728",
                          opacity=opacity, layer="below", line_width=0)
            in_blk = False

def ensure_kaleido() -> bool:
    """
    Try to import kaleido. If missing and the platform allows, attempt a single
    runtime install. Return True if available, False otherwise.
    """
    try:
        importlib.import_module("kaleido")
        return True
    except Exception:
        pass

    # Avoid repeated attempts in the same session
    if st.session_state.get("_kaleido_attempted", False):
        return False

    st.session_state["_kaleido_attempted"] = True

    # Some managed hosts allow runtime pip; if it fails we just fall back gracefully
    try:
        with st.spinner("Installing 'kaleido' once for PNG export…"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido==0.2.1"])
        importlib.import_module("kaleido")
        return True
    except Exception:
        return False


def download_png(fig, label: str, filename: str, *, width: int = 2200, height: int = 900, scale: int = 6):
    """
    High-res PNG export using Kaleido if available. If not, show a clean fallback
    message to use the Plotly camera icon.
    """
    try:
        if ensure_kaleido():
            png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")
            st.download_button(label, png_bytes, file_name=filename, mime="image/png", use_container_width=True)
        else:
            with st.expander("PNG download (Kaleido not available)"):
                st.info(
                    "PNG export needs the 'kaleido' renderer. "
                    "On this host it isn’t available right now. "
                    "Use the camera icon in the chart toolbar to save a PNG, "
                    "or deploy with `runtime.txt` = `3.11` and `kaleido==0.2.1` in `requirements.txt`."
                )
    except Exception as e:
        with st.expander("PNG download (renderer error)"):
            st.warning(
                f"Couldn’t render PNG with Kaleido.\n\n"
                f"Details: {e}\n\n"
                "Use the camera icon in the chart toolbar to save a PNG for now."
            )

# ===================== UI =====================

st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")

st.sidebar.header("Controls")
zoom_years = st.sidebar.slider("Zoom window (years)", 1, 10, 3, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Regime knobs")

n_components = st.sidebar.selectbox("HMM states (n_components)", [2, 3, 4], index=2)
ema_span      = st.sidebar.slider("EMA smoothing of bear prob (ema_span)", 5, 50, 20)
bear_enter    = st.sidebar.slider("Bear enter threshold", 0.50, 0.99, 0.80, 0.01)
bear_exit     = st.sidebar.slider("Bear exit threshold", 0.10, 0.90, 0.55, 0.01)
mom_thr       = st.sidebar.slider("Trend weakness (mom_threshold)", 0.00, 0.10, 0.03, 0.005)
dd_thr        = st.sidebar.slider("Drawdown confirm (ddown_threshold)", 0.00, 0.40, 0.15, 0.01)
confirm_d     = st.sidebar.slider("Confirm days", 1, 15, 7, 1)

bull_mom      = st.sidebar.slider("Bull trend (bull_mom_threshold)", 0.00, 0.05, 0.01, 0.005)
bull_dd_ex    = st.sidebar.slider("Bull dd exit (bull_ddown_exit)", 0.00, 0.20, 0.06, 0.01)
confirm_db    = st.sidebar.slider("Confirm days (bull)", 1, 10, 3, 1)

min_bear_run  = st.sidebar.slider("Min bear run (days)", 1, 30, 15, 1)
min_bull_run  = st.sidebar.slider("Min bull run (days)", 1, 30, 5, 1)

direction_gate    = st.sidebar.checkbox("Use direction gate", value=True)
entry_lb          = st.sidebar.number_input("Entry return lookback (days)", 1, 60, 10)
entry_ret         = st.sidebar.number_input("Entry return thresh (≤ negative)", -0.20, 0.00, -0.01, step=0.005, format="%.3f")
entry_dd          = st.sidebar.number_input("Entry drawdown thresh (≤ negative)", -0.50, 0.00, -0.03, step=0.005, format="%.3f")
trend_gate        = st.sidebar.checkbox("Require price < EMA100 at entry", value=True)
trend_exit_cross  = st.sidebar.checkbox("Exit bear when price crosses above EMA100", value=True)
bear_profit_exit  = st.sidebar.number_input("Bear profit exit (bounce % from entry)", 0.00, 0.30, 0.05, step=0.005, format="%.3f")

# ===================== Price charts (always render) =====================

with st.spinner("Fetching TSLA full price history for charts…"):
    px_full_price = fetch_tsla_full_history()

if px_full_price.empty:
    st.error("No TSLA data returned by Yahoo right now. Try 'Rerun' (↻) in the app header.")
    st.stop()

px_full = add_emas(px_full_price, fast=20, slow=100)
px_zoom = last_years(px_full, zoom_years)

full_years = (px_full.index.max() - px_full.index.min()).days / 365.25
st.caption(
    f"Full-history price with EMAs (IPO→today) and last-{zoom_years}-years view. "
    f"Charts use direct Yahoo full history; regimes are computed separately. "
    f"Diagnostic — full span: {full_years:,.1f}y; zoom window: {zoom_years}y."
)

# -------- Chart 1: IPO→today (from full Yahoo series) --------
fig1 = plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)")
st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
download_png(fig1, "Download full-history chart (high-res)", "tsla_full_history.png")

# -------- Chart 2: last N years (slice of the full series) ---
fig2 = plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)")
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
download_png(fig2, "Download zoom chart (high-res)", "tsla_last_years.png")

# ===================== Regime plot (runs after price charts) =====================

with st.spinner("Computing regimes (HMM + rules)…"):
    # IMPORTANT: pass end=None (NOT 'today') so the pipeline doesn’t error
    df_reg, _ = detect_regimes(
        ticker="TSLA",
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
        start="2010-06-29",
        end=None,  # <— critical
    )

# Regime chart = last N years only
fig3 = plot_close_emas(px_zoom, f"TSLA — Regimes (last {zoom_years} years; light=candidate, dark=confirmed)", h=480)
try:
    add_bear_shading(fig3, df_reg, px_zoom, "bear_candidate", opacity=0.12)
    add_bear_shading(fig3, df_reg, px_zoom, "bear_confirm",   opacity=0.30)
except Exception as e:
    st.warning(f"Could not shade regimes: {e}")
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
download_png(fig3, "Download regimes chart (high-res)", "tsla_regimes_last_years.png")


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
