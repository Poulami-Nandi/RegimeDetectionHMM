# streamlit_app/Home.py
from __future__ import annotations

# ===== stdlib =====
import inspect
from functools import wraps
from pathlib import Path
import importlib
import subprocess
import sys

# ===== third-party =====
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --- make 'src' importable on Streamlit Cloud ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Friendly failure if src isn’t importable
try:
    from src.regime_detection import detect_regimes
except Exception as e:
    st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
    st.error(
        "Could not import pipeline module `src.regime_detection`.\n\n"
        "Ensure **src/** exists at the repo root and contains `__init__.py`.\n"
        f"Repo root seen by app: `{REPO_ROOT}`\n\n"
        f"Import error: {e}"
    )
    st.stop()

st.set_page_config(
    page_title="TSLA Regime Detection — HMM + Rules",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== Small utils =====================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tsla_full_history() -> pd.DataFrame:
    """Full-history TSLA Close (auto-adjusted)."""
    try:
        df = yf.download("TSLA", period="max", auto_adjust=True, progress=False)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Close"]].dropna()
    except Exception as e:
        st.error(f"Yahoo fetch failed: {e}")
        return pd.DataFrame(columns=["Close"])

def add_emas(df: pd.DataFrame, fast=20, slow=100) -> pd.DataFrame:
    out = df.copy()
    if not out.empty and "Close" in out:
        out["EMA20"] = out["Close"].ewm(span=fast, adjust=False).mean()
        out["EMA100"] = out["Close"].ewm(span=slow, adjust=False).mean()
    return out

def last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()

def ensure_kaleido() -> bool:
    """Try to import kaleido (for PNG export). Attempt runtime install once."""
    try:
        importlib.import_module("kaleido")
        return True
    except Exception:
        pass
    if st.session_state.get("_kaleido_attempted", False):
        return False
    st.session_state["_kaleido_attempted"] = True
    try:
        with st.spinner("Installing 'kaleido' for PNG export…"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido==0.2.1"])
        importlib.import_module("kaleido")
        return True
    except Exception:
        return False

def download_png(fig, label: str, filename: str, *, width: int = 2200, height: int = 900, scale: int = 6):
    try:
        if ensure_kaleido():
            png = fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")
            st.download_button(label, png, file_name=filename, mime="image/png", use_container_width=True)
        else:
            st.info("PNG export needs `kaleido`. Use the camera icon in the chart toolbar.")
    except Exception:
        st.info("PNG render failed. Use the camera icon in the chart toolbar.")

def _ensure_emas_local(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee lowercase ema20/ema100 exist; copy from uppercase or compute."""
    if df is None or df.empty:
        return df
    if "EMA20" in df.columns and "ema20" not in df.columns:
        df["ema20"] = df["EMA20"]
    if "EMA100" in df.columns and "ema100" not in df.columns:
        df["ema100"] = df["EMA100"]
    if "Close" in df.columns:
        if "ema20" not in df.columns:
            df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "ema100" not in df.columns:
            df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df

def _segments(index, mask_bool):
    """Return [(start, end), ...] for contiguous True runs in mask_bool."""
    out, start = [], None
    arr = mask_bool.values if hasattr(mask_bool, "values") else mask_bool
    for i, v in enumerate(arr):
        v = bool(v)
        if v and start is None:
            start = index[i]
        elif (not v) and (start is not None):
            out.append((start, index[i-1])); start = None
    if start is not None:
        out.append((start, index[-1]))
    return out

def _add_vbands(fig, idx, mask_bool, color, opacity, y0, y1):
    for s, e in _segments(idx, mask_bool):
        fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=opacity,
                      line_width=0, layer="below", y0=y0, y1=y1)

# ===================== UI =====================
st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")

# Sidebar (TSLA fixed)
st.sidebar.markdown("### Controls (fixed to TSLA)")
ticker = "TSLA"

st.sidebar.markdown("### Regime knobs (defaults match your Colab)")
n_components        = st.sidebar.number_input("HMM states (n_components)", 2, 6, value=4, step=1)
k_forward           = st.sidebar.slider("k_forward (days ahead for label)", 1, 20, value=10, step=1)
ema_span            = st.sidebar.slider("EMA smoothing of bear prob (ema_span)", 5, 60, value=20, step=1)
bear_enter          = st.sidebar.slider("Bear enter threshold", 0.50, 0.99, value=0.80, step=0.01)
bear_exit           = st.sidebar.slider("Bear exit threshold", 0.00, 0.95, value=0.55, step=0.01)
min_bear_run        = st.sidebar.slider("Min bear run (days)", 1, 60, value=15, step=1)
min_bull_run        = st.sidebar.slider("Min bull run (days)", 1, 60, value=5, step=1)
mom_threshold       = st.sidebar.slider("Trend weakness (mom_threshold)", 0.00, 0.10, value=0.03, step=0.001)
ddown_threshold     = st.sidebar.slider("Drawdown confirm (ddown_threshold)", 0.00, 0.30, value=0.15, step=0.005)
confirm_days        = st.sidebar.slider("Confirm days (bear)", 0, 20, value=7, step=1)
bull_mom_threshold  = st.sidebar.slider("Bull trend (bull_mom_threshold)", 0.00, 0.05, value=0.01, step=0.001)
bull_ddown_exit     = st.sidebar.slider("Bull dd exit (bull_ddown_exit)", 0.00, 0.20, value=0.06, step=0.005)
confirm_days_bull   = st.sidebar.slider("Confirm days (bull)", 0, 10, value=3, step=1)
direction_gate      = st.sidebar.checkbox("Directional gate", value=True)
trend_gate          = st.sidebar.checkbox("Trend gate", value=True)
strict              = st.sidebar.checkbox("Strict confirmation", value=False)
entry_ret_lookback  = st.sidebar.slider("Entry return lookback (days)", 1, 30, value=10, step=1)
entry_ret_thresh    = st.sidebar.slider("Entry return threshold", -0.05, 0.05, value=-0.01, step=0.001)
entry_ddown_thresh  = st.sidebar.slider("Entry drawdown threshold", -0.10, 0.10, value=-0.03, step=0.001)
bear_profit_exit    = st.sidebar.slider("Bear profit exit", 0.00, 0.20, value=0.05, step=0.005)
zoom_years          = st.sidebar.slider("Zoom window (years)", 1, 10, value=3, step=1)

# ===================== Price charts (always render) =====================
with st.spinner("Fetching TSLA full price history…"):
    px_full_price = fetch_tsla_full_history()
if px_full_price.empty:
    st.error("No TSLA data returned by Yahoo right now. Try rerunning.")
    st.stop()

px_full = add_emas(px_full_price, fast=20, slow=100)
px_zoom = last_years(px_full, zoom_years)

full_years = (px_full.index.max() - px_full.index.min()).days / 365.25
st.caption(
    f"Full-history price with EMAs (IPO→today) and last-{zoom_years}-years view. "
    f"Charts use direct Yahoo full history; regimes are computed separately. "
    f"Diagnostic — full span: {full_years:,.1f}y; zoom window: {zoom_years}y."
)

def plot_close_emas(df: pd.DataFrame, title: str, h=440) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                 mode="lines", line=dict(width=2.2, color="#111")))
        if "EMA20" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20",
                                     mode="lines", line=dict(width=1.7, color="#ff7f0e")))
        if "EMA100" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA100"], name="EMA100",
                                     mode="lines", line=dict(width=1.7, color="#2ca02c")))
    fig.update_layout(
        title=dict(text=title, pad=dict(b=28)),
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=10, t=78, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig

fig1 = plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)")
st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
download_png(fig1, "Download full-history chart (PNG)", "tsla_full_history.png")

fig2 = plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)")
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
download_png(fig2, "Download zoom chart (PNG)", "tsla_last_years.png")

# ======================= RUN PIPELINE (TSLA only) =======================
def _call_detect_regimes_flexible(func, **vals):
    """Map our values to whatever parameter names `detect_regimes` supports."""
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    out = {}
    def put(names, value):
        for name in names:
            if name in params:
                out[name] = value; return
    put(["ticker"],                 vals.get("ticker"))
    put(["start"],                  vals.get("start"))
    put(["end"],                    vals.get("end"))
    put(["n_components"],           vals.get("n_components"))
    put(["k_forward", "k_fwd"],     vals.get("k_forward"))
    put(["ema_span", "ema"],        vals.get("ema_span"))
    put(["prob_threshold", "bear_enter", "enter_prob", "enter_threshold"], vals.get("bear_enter"))
    put(["prob_exit", "bear_exit", "exit_prob", "exit_threshold"],        vals.get("bear_exit"))
    put(["min_run", "min_bear_run"],                vals.get("min_bear_run"))
    put(["min_bull_run"],                           vals.get("min_bull_run"))
    put(["mom_threshold", "mom_thr"],               vals.get("mom_threshold"))
    put(["ddown_threshold", "dd_thr"],              vals.get("ddown_threshold"))
    put(["confirm_days", "confirm_bear"],           vals.get("confirm_days"))
    put(["bull_mom_threshold", "bull_mom_thr"],     vals.get("bull_mom_threshold"))
    put(["bull_ddown_exit", "bull_dd_exit"],        vals.get("bull_ddown_exit"))
    put(["confirm_days_bull", "confirm_bull"],      vals.get("confirm_days_bull"))
    put(["direction_gate"],                         vals.get("direction_gate"))
    put(["trend_gate"],                             vals.get("trend_gate"))
    put(["entry_ret_lookback", "lbk", "lookback"],  vals.get("entry_ret_lookback"))
    put(["entry_ret_thresh", "entry_ret_thr"],      vals.get("entry_ret_thresh"))
    put(["entry_ddown_thresh", "entry_dd_thr"],     vals.get("entry_ddown_thresh"))
    put(["bear_profit_exit", "profit_exit"],        vals.get("bear_profit_exit"))
    put(["strict"],                                 vals.get("strict"))
    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)

# (optional) alias wrapper so direct calls work too
_ALIAS_MAP = {
    "bear_enter":         ["prob_threshold", "enter_prob", "enter_threshold"],
    "bear_exit":          ["prob_exit", "exit_prob", "exit_threshold"],
    "k_forward":          ["k_fwd", "k"],
    "ema_span":           ["ema", "ema_smooth"],
    "min_bear_run":       ["min_run"],
    "ddown_threshold":    ["dd_thr"],
    "confirm_days":       ["confirm_bear", "confirm"],
    "bull_mom_threshold": ["bull_mom_thr"],
    "bull_ddown_exit":    ["bull_dd_exit", "bull_ddown_thr"],
    "confirm_days_bull":  ["confirm_bull"],
    "entry_ret_lookback": ["lbk", "lookback"],
    "entry_ret_thresh":   ["entry_ret_thr"],
    "entry_ddown_thresh": ["entry_dd_thr"],
    "bear_profit_exit":   ["profit_exit"],
}
def _wrap_detect_regimes(func):
    sig_names = set(inspect.signature(func).parameters.keys())
    def _map_kwargs(kwargs):
        out = {}
        for k, v in kwargs.items():
            if v is None: continue
            if k in sig_names:
                out[k] = v
            else:
                for alt in _ALIAS_MAP.get(k, ()):
                    if alt in sig_names:
                        out[alt] = v; break
        return out
    def wrapper(*args, **kwargs):
        if args:  # passthrough if positional used
            return func(*args, **kwargs)
        return func(**_map_kwargs(kwargs))
    return wrapper
detect_regimes = _wrap_detect_regimes(detect_regimes)

df, _ = _call_detect_regimes_flexible(
    detect_regimes,
    ticker="TSLA", start="2000-01-01", end="today",
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

# Ensure EMAs exist for plotting
if "ema20" not in df.columns:
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
if "ema100" not in df.columns:
    df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()

# Last N years for regime view
px_zoom = last_years(df, zoom_years)
px_zoom = _ensure_emas_local(px_zoom)

# Regime masks (strict booleans)
bear_cand = pd.Series(px_zoom.get("bear_candidate", False), index=px_zoom.index).fillna(False).astype(bool)
bear_conf = pd.Series(px_zoom.get("bear_confirm",   False), index=px_zoom.index).fillna(False).astype(bool)
cand_only = bear_cand & (~bear_conf)

# ===== Plot regimes (bear: light=candidate, dark=confirmed) =====
ymin = float(px_zoom["Close"].min()) * 0.95
ymax = float(px_zoom["Close"].max()) * 1.05

fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["Close"],  name="Close",
                             mode="lines", line=dict(width=2.0, color="#111")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.6, color="#ff7f0e")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.6, color="#2ca02c")))

# background shading
_add_vbands(fig_reg, px_zoom.index, cand_only.values, "crimson", 0.12, ymin, ymax)
_add_vbands(fig_reg, px_zoom.index, bear_conf.values, "crimson", 0.30, ymin, ymax)

params_str = (
    f"k_fwd={k_forward}, EMA={ema_span}, enter={bear_enter:.2f}, exit={bear_exit:.2f}, "
    f"min_bear={min_bear_run}, min_bull={min_bull_run}, mom_thr={mom_threshold:.2f}, "
    f"dd_thr={ddown_threshold:.2f}, bull_mom_thr={bull_mom_threshold:.2f}, "
    f"bull_dd_exit={bull_ddown_exit:.2f}, confirm_bear={confirm_days}, "
    f"confirm_bull={confirm_days_bull}, dir_gate={direction_gate}, "
    f"lbk={entry_ret_lookback}, entry_ret_thr={entry_ret_thresh:.2f}, "
    f"entry_dd_thr={entry_ddown_thresh:.2f}, trend_gate={trend_gate}, "
    f"profit_exit={bear_profit_exit:.2f}, strict={strict}"
)

fig_reg.update_layout(
    title=dict(text=f"TSLA — Regimes (last {zoom_years} years; light=candidate, dark=confirmed)<br>"
                    f"<sup>{params_str}</sup>", pad=dict(b=26)),
    template="plotly_white",
    height=520,
    margin=dict(l=10, r=10, t=90, b=10),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", range=[ymin, ymax]),
    hovermode="x unified",
)
st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")

try:
    st.download_button(
        "Download regimes chart (PNG)",
        data=fig_reg.to_image(format="png", scale=3),
        file_name=f"{ticker}_regimes_last{zoom_years}y.png",
        mime="image/png",
    )
except Exception:
    st.info("PNG export needs `kaleido`. Use the chart camera icon if unavailable.")

# ================= Parameter explainer =================
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
- **`bear_enter`** — start a **bear candidate** when smoothed bear-prob ≥ this (e.g., 0.80).  
- **`bear_exit`** — leave bear when smoothed bear-prob ≤ this (keep well below enter; e.g., 0.55).
"""
)

st.markdown("### Confirmations & filters")
st.markdown(
"""
- **`mom_threshold`** — EMA20 must sit **below** EMA100 by this fraction to *confirm* a bear (e.g., 3%).  
- **`ddown_threshold`** — price must be ≥ this far below a recent peak (e.g., 15%).  
- **`confirm_days`** — how long weakness must persist (e.g., 7 days).  
- **`min_bear_run`/`min_bull_run`** — drop tiny islands shorter than these.  
- **`bull_mom_threshold`, `confirm_days_bull`** — analogous confirmations for bull pockets.  
- **`bull_ddown_exit`** — if drawdown heals near the peak, exit bear early.
"""
)

st.markdown("### Gates & quick tuning")
st.markdown(
"""
- **`direction_gate`** and **`trend_gate`** further restrict bear entries.  
- Too many false bears? Raise `bear_enter`, `mom_threshold`, `ddown_threshold`, `confirm_days`.  
- Bear exits too late? Lower `bear_exit` a touch, or raise `bull_ddown_exit`.
"""
)

st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
