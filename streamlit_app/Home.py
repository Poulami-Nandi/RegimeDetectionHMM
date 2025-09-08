# streamlit_app/Home.py  — crisp, zoomable charts (Plotly) + hi-res downloads
import io
import inspect
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")
st.caption("Full-history price with EMAs (IPO→today) + last-N-years regime view. Data via Yahoo Finance at runtime.")

# ---------------- Import your pipeline ----------------
try:
    from src.regime_detection import detect_regimes
    PIPELINE_IMPORT_ERROR = None
except Exception as e:
    detect_regimes = None
    PIPELINE_IMPORT_ERROR = e

# ---------------- Sidebar controls ----------------
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
    zoom_years = st.slider("Zoom window (last N years)", 1, 5, 3, 1)
    download_scale = st.slider("Download image scale (higher = more pixels)", 2, 8, 5, 1)

st.sidebar.info("ℹ️ Light red = candidate bear (probability crossed ENTER). Dark red = confirmed bear (weak trend/drawdown persisted).")

# ---------------- Guardrail on import failure ----------------
if detect_regimes is None:
    st.error(
        "Couldn't import pipeline: `from src.regime_detection import detect_regimes` failed.\n\n"
        f"Error: {PIPELINE_IMPORT_ERROR}\n"
        "Ensure src/regime_detection.py defines detect_regimes()."
    )
    st.stop()

ticker = "TSLA"

# ---------------- Safe pipeline call ----------------
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
        return_debug=True,
        start="2010-01-01",
        end="today",
    )

    # Back-compat synonyms
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
    if "full_history" in params:
        pref["full_history"] = True

    filtered = {k: v for k, v in pref.items() if k in params}
    try:
        return detect_regimes(**filtered)
    except TypeError as e:
        st.warning("Falling back to minimal call due to signature mismatch: " + str(e))
        return detect_regimes(ticker=ticker, n_components=int(n_components))

with st.spinner("Downloading data & computing regimes..."):
    df, model = call_detect_regimes_safely()

if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Pipeline returned an empty DataFrame. Try different settings.")
    st.stop()

# Ensure EMAs exist
if "sma20" not in df.columns:
    df["sma20"] = df["Close"].ewm(span=20, adjust=False).mean()
if "sma100" not in df.columns:
    df["sma100"] = df["Close"].ewm(span=100, adjust=False).mean()

# ---------------- Independent FULL HISTORY for chart A ----------------
@st.cache_data(ttl=6*3600, show_spinner=False)
def fetch_full_history(tkr: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tkr, period="max", interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = raw[["Close"]].copy()
    out.index = pd.to_datetime(out.index)
    out["sma20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["sma100"] = out["Close"].ewm(span=100, adjust=False).mean()
    return out.dropna()

df_full = fetch_full_history(ticker)

span_full = (df_full.index.max() - df_full.index.min()).days/365.25 if not df_full.empty else 0.0
span_pipe = (df.index.max() - df.index.min()).days/365.25
st.caption(f"Diagnostic — full-history span: {span_full:.1f}y; pipeline span: {span_pipe:.1f}y; zoom window: {int(zoom_years)}y")

if df_full.empty:
    st.warning("Couldn't fetch full history from Yahoo. Showing pipeline window in the first chart too.")
    df_full = df.copy()

# ---------------- Slice last N years ----------------
cutoff = df.index.max() - pd.DateOffset(years=int(zoom_years))
dfz = df[df.index >= cutoff].copy()

# ---------------- Plot helpers (Plotly) ----------------
import plotly.graph_objects as go

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "TSLA_chart",
        "width": 2400,      # ~4K wide
        "height": 1200,
        "scale": 1          # users can also use the slider below for bigger exports
    }
}

def figure_base(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.15)"),
        template="plotly_white",
    )
    return fig

def add_close_ema(fig, frame: pd.DataFrame, name_prefix="TSLA"):
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["Close"], name=f"{name_prefix} Close",
        mode="lines", line=dict(width=1.6, color="black")
    ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["sma20"], name="EMA20 (fast)",
        mode="lines", line=dict(width=1.2)
    ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["sma100"], name="EMA100 (slow)",
        mode="lines", line=dict(width=1.2)
    ))

def add_vrect_segments(fig, dates, mask, rgba, name_for_legend):
    """Shade contiguous True regions of mask with vrects; add a single dummy trace for legend."""
    if mask is None or not bool(mask.any()):
        return
    # legend proxy
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color=rgba.replace("0.2", "1.0"), width=10),
        name=name_for_legend,
        showlegend=True
    ))
    in_run, start = False, None
    vals = mask.values
    for i, on in enumerate(vals):
        if on and not in_run:
            in_run, start = True, dates[i]
        last = (i == len(vals) - 1)
        if in_run and (not on or last):
            end = dates[i]
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=rgba, line_width=0, layer="below"
            )
            in_run, start = False, None

def hi_res_download_buttons(fig, base_filename: str, scale: int):
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", width=2400*scale//5, height=1200*scale//5, scale=5)
        st.download_button(
            f"⬇️ Download PNG ({(2400*scale//5)}x{(1200*scale//5)} @x5)",
            data=png_bytes, file_name=f"{base_filename}_{(2400*scale//5)}x{(1200*scale//5)}.png",
            mime="image/png"
        )
    except Exception as e:
        st.caption("PNG download requires 'kaleido'. If missing on the host, use the camera icon in the chart toolbar instead.")

# ---------------- Plot A: Full history Close + EMAs (IPO→today) ----------------
fig_full = figure_base("TSLA — Close with EMA20 / EMA100 (IPO → today)")
add_close_ema(fig_full, df_full)
st.plotly_chart(fig_full, use_container_width=True, config=PLOT_CONFIG)
with st.expander("Download full-history chart (high resolution)"):
    hi_res_download_buttons(fig_full, "TSLA_full_history", download_scale)

# ---------------- Plot B: Close + EMAs (last N years) ----------------
fig_last = figure_base(f"TSLA — Close with EMA20 / EMA100 (last {int(zoom_years)} years)")
add_close_ema(fig_last, dfz)
st.plotly_chart(fig_last, use_container_width=True, config=PLOT_CONFIG)
with st.expander("Download last-N-years chart (high resolution)"):
    hi_res_download_buttons(fig_last, f"TSLA_close_ema_last{int(zoom_years)}y", download_scale)

# ---------------- Plot C: Regimes (last N years) ----------------
fig_reg = figure_base(f"TSLA — Regimes (last {int(zoom_years)} years)")
add_close_ema(fig_reg, dfz)  # price + EMAs on top

# Build candidate/confirmed masks
conf_z = (dfz["regime"] == 1) if "regime" in dfz.columns else None
cand_z = None
if "p_bear_ema" in dfz.columns:
    cand_z = dfz["p_bear_ema"] >= float(bear_enter)
    # exclude confirmed portions from candidate, if present
    if "bear_confirm" in dfz.columns:
        cand_z = cand_z & (~dfz["bear_confirm"].astype(bool))
    if conf_z is not None:
        cand_z = cand_z & (~conf_z)
elif "bear_candidate" in dfz.columns:
    cand_z = dfz["bear_candidate"].astype(bool)

# Shade
if cand_z is not None and cand_z.any():
    add_vrect_segments(fig_reg, dfz.index, cand_z, "rgba(255,0,0,0.22)", "Bear (candidate)")
if conf_z is not None and conf_z.any():
    add_vrect_segments(fig_reg, dfz.index, conf_z, "rgba(255,0,0,0.45)", "Bear (confirmed)")

st.plotly_chart(fig_reg, use_container_width=True, config=PLOT_CONFIG)
with st.expander("Download regimes chart (high resolution)"):
    hi_res_download_buttons(fig_reg, f"TSLA_regimes_last{int(zoom_years)}y", download_scale)

# ---------------- Explainer ----------------
st.markdown("---")
st.markdown("## Parameter explainer")
st.markdown(
    """
**Acronyms**
- **HMM:** Hidden Markov Model — unsupervised model that assigns a hidden “state” to each day.
- **EMA:** Exponential Moving Average — recent prices weighted more than older prices.

**General**
- **n_components** — number of hidden states the HMM can use.
- **ema_span** — smoothing for daily bear probability; bigger = steadier.
- **k_forward** — look-ahead days used only to *name* states (bullish vs bearish behavior).

**Hysteresis thresholds**
- **bear_enter** — create a *bear candidate* when smoothed bear probability rises above this.
- **bear_exit** — exit bear when it falls below this (prevents flip-flops).

**Bear confirmations (turn candidate → confirmed)**
- **mom_threshold** — EMA20 under EMA100 by ~X of price (trend weakness).
- **ddown_threshold** — price ~X below a recent peak (drawdown).
- **confirm_days** — number of consecutive days those must hold.

**Bull confirms & early exits (to end bear)**
- **bull_mom_threshold** — EMA20 above EMA100 by ~X.
- **bull_ddown_exit** — drawdown recovered toward peak (e.g., within 6%).
- **confirm_days_bull** — consecutive days those must hold.
- **bear_profit_exit** — if price rallies +X% from the bear entry, exit quickly even if probability lags.

**Gates & cleanup**
- **direction_gate** — only allow bear entry if last L-day return ≤ threshold *and* current drawdown ≤ threshold.
- **trend_gate** — require price/EMA20 under EMA100 at entry.
- **trend_exit_cross** — exit bear on cross up of price/EMA20 over EMA100.
- **min_bear_run / min_bull_run** — remove tiny islands for clarity.

**Auto thresholds (optional)**
- **auto_thresholds** — learn enter/exit from recent years to hit **bear_target** share.
- **auto_window_years** — how far back to learn from.
- **min_gap** — minimum spacing between enter and exit so they aren’t equal.

**Tip for downloads:** Use the camera icon in each chart (modebar) to export PNG.  
These charts are SVG in the browser (vector), so zoom stays crisp while presenting.
"""
)
st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
