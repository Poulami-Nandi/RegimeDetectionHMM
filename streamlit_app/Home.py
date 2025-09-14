# streamlit_app/Home.py
from __future__ import annotations
import inspect
from functools import wraps

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import io
import sys
import importlib
import subprocess

# --- make 'src' importable on Streamlit Cloud ---
import sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (one level above streamlit_app)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---- add near the top of Home.py (after other imports) ----
import io, zipfile
import pandas as pd

def _csv_bytes(obj, float_format="%.6f") -> bytes:
    """
    Convert a Series/DataFrame (or something convertible) to CSV bytes.
    Safe on None/missing, keeps index, and formats dates/floats.
    """
    if obj is None:
        return b""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if not isinstance(obj, pd.DataFrame):
        try:
            obj = pd.DataFrame(obj)
        except Exception:
            return str(obj).encode("utf-8")

    return obj.to_csv(
        index=True,
        date_format="%Y-%m-%d",
        float_format=float_format,
    ).encode("utf-8")


# Friendly failure if src still isn't importable
try:
    from src.regime_detection import detect_regimes
except Exception as e:
    st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
    st.error(
        "Could not import pipeline module `src.regime_detection`.\n\n"
        "Fix by ensuring **src/** exists at the repo root and contains an `__init__.py` file.\n"
        f"Repo root seen by app: `{REPO_ROOT}`\n\n"
        f"Underlying import error: {e}"
    )
    st.stop()

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
        title=dict(text=title, pad=dict(b=32)),
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.02,
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)"),
    )

    return fig

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

# ======================= SIDEBAR: fixed TSLA + default knobs =======================
import datetime as _dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.sidebar.markdown("### Controls (fixed to TSLA)")
ticker = "TSLA"  # <- locked

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

# ===================== Regime masks (robust) + CSV save + shading =====================
from pathlib import Path
import numpy as np

def _safe_bool(df: pd.DataFrame, name: str) -> pd.Series:
    s = df.get(name, pd.Series(False, index=df.index))
    return pd.Series(s, index=df.index).fillna(False).astype(bool)

def _segments_index(index: pd.DatetimeIndex, mask: pd.Series | np.ndarray, min_len: int = 1):
    """Return [(start, end), ...] for contiguous True runs; min_len measured in bars."""
    mask = np.asarray(mask, dtype=bool)
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = index[i]
        if (not v or i == len(mask) - 1) and start is not None:
            end = index[i] if v and i == len(mask) - 1 else index[i - 1]
            if (index.get_indexer([end])[0] - index.get_indexer([start])[0] + 1) >= min_len:
                out.append((start, end))
            start = None
    return out

# ---- try to read masks from pipeline output ----
bear_cand = _safe_bool(px_zoom, "bear_candidate")
bear_conf = _safe_bool(px_zoom, "bear_confirm")
have_pipeline_masks = bear_cand.any() or bear_conf.any()

# ---- fallback masks (when pipeline didn’t populate columns) ----
if not have_pipeline_masks:
    # Drawdown from rolling max (lookback = 252 trading days ~ 1y)
    roll_max = px_zoom["Close"].cummax()
    ddown = 1.0 - (px_zoom["Close"] / roll_max)                 # 0 … 1
    mom_gap = (px_zoom["ema20"] - px_zoom["ema100"]) / px_zoom["ema100"]

    # Candidate bear: weak momentum OR drawdown elevated
    bear_cand = ((mom_gap < 0) | (ddown > max(0.05, ddown_threshold))).astype(bool)

    # Confirmed bear: both weak momentum and drawdown beyond threshold, with persistence
    raw_conf = ((mom_gap < -abs(mom_threshold)) & (ddown > ddown_threshold)).astype(int)
    # persistence (confirm_days); if confirm_days=0 this is a no-op
    if confirm_days > 0:
        conf_roll = raw_conf.rolling(confirm_days, min_periods=confirm_days).sum()
        bear_conf = (conf_roll >= confirm_days).astype(bool).reindex(px_zoom.index).fillna(False)
    else:
        bear_conf = raw_conf.astype(bool)

# ---- bull masks: prefer pipeline, otherwise infer from bear masks / momentum ----
bull_cand = _safe_bool(px_zoom, "bull_candidate")
bull_conf = _safe_bool(px_zoom, "bull_confirm")

if not (bull_cand.any() or bull_conf.any()):
    # If pipeline didn’t provide bull_*, infer them
    bull_conf = (~bear_conf) & (px_zoom["ema20"] > px_zoom["ema100"])
    # candidate bull = non-confirmed area leaning bullish (but not yet confirmed)
    bull_cand = (~bear_conf) & (~bull_conf) & (px_zoom["ema20"] >= px_zoom["ema100"])

# Candidate-only (to avoid double shading with confirmed)
bull_cand_only = bull_cand & (~bull_conf)
bear_cand_only = bear_cand & (~bear_conf)

# ---- Save confirmed segments (last Ny window) to CSV and show a preview ----
def _pick(v: pd.Series, t):
    try:
        return float(v.loc[t])
    except Exception:
        return np.nan

rows = []
for label, mask in (("bull_confirm", bull_conf), ("bear_confirm", bear_conf)):
    for s, e in _segments_index(px_zoom.index, mask, min_len=1):
        entry_p = _pick(px_zoom["Close"], s)
        exit_p  = _pick(px_zoom["Close"], e)
        ret = (exit_p / entry_p - 1.0) if (np.isfinite(entry_p) and np.isfinite(exit_p) and entry_p != 0) else np.nan
        rows.append({
            "type": label,
            "start": s.strftime("%Y-%m-%d"),
            "end":   e.strftime("%Y-%m-%d"),
            "entry_price": entry_p,
            "exit_price":  exit_p,
            "days": (e - s).days + 1,
            "return": ret,
        })

seg_df = pd.DataFrame(rows, columns=["type","start","end","entry_price","exit_price","days","return"])
reports_dir = Path(REPO_ROOT, "reports")
reports_dir.mkdir(parents=True, exist_ok=True)
csv_path = reports_dir / "confirmed_segments_last3y.csv"
seg_df.to_csv(csv_path, index=False)

with st.expander("Confirmed regime segments (saved to CSV)", expanded=False):
    st.write(f"Saved to: `{csv_path}`")
    st.dataframe(seg_df, use_container_width=True)
    st.download_button("Download CSV", data=seg_df.to_csv(index=False).encode("utf-8"),
                       file_name="confirmed_segments_last3y.csv", mime="text/csv")

# ---- 4-color shading on the regimes chart ----
def _add_color_bands(fig: go.Figure, idx: pd.DatetimeIndex, mask: pd.Series, color: str, opacity: float):
    for s, e in _segments_index(idx, mask):
        fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=opacity, line_width=0, layer="below")

ymin = float(px_zoom["Close"].min()) * 0.95
ymax = float(px_zoom["Close"].max()) * 1.05

# If you created fig_reg above, apply the bands to that figure:
#   light green  = bull candidate; dark green  = bull confirmed
#   light red    = bear candidate; dark red    = bear confirmed
_add_color_bands(fig_reg, px_zoom.index, bull_cand_only, "#2ca02c", 0.12)
_add_color_bands(fig_reg, px_zoom.index, bull_conf,      "#2ca02c", 0.30)
_add_color_bands(fig_reg, px_zoom.index, bear_cand_only, "#d62728", 0.12)
_add_color_bands(fig_reg, px_zoom.index, bear_conf,      "#d62728", 0.30)
# -------- Chart 1: IPO→today (from full Yahoo series) --------
fig1 = plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)")
st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
download_png(fig1, "Download full-history chart (high-res)", "tsla_full_history.png")

# -------- Chart 2: last N years (slice of the full series) ---
fig2 = plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)")
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
download_png(fig2, "Download zoom chart (high-res)", "tsla_last_years.png")

# ===================== Regime plot (runs after price charts) =====================

def _call_detect_regimes_flexible(func, **vals):
    """
    Map our concept values to whatever parameter names `detect_regimes` supports.
    Returns (df, model) where model may be None if your function returns only df.
    """
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())

    out = {}

    def put(names, value):
        """Assign `value` to the first name in `names` that exists in func params."""
        for name in names:
            if name in params:
                out[name] = value
                return

    # always try these core ones
    put(["ticker"],                 vals.get("ticker"))
    put(["start"],                  vals.get("start"))
    put(["end"],                    vals.get("end"))
    put(["n_components"],           vals.get("n_components"))

    # common aliases across your versions
    put(["k_forward", "k_fwd"],                         vals.get("k_forward"))
    put(["ema_span", "ema"],                            vals.get("ema_span"))
    put(["prob_threshold", "bear_enter", "enter_prob",
         "enter_threshold"],                            vals.get("bear_enter"))
    put(["prob_exit", "bear_exit", "exit_prob",
         "exit_threshold"],                             vals.get("bear_exit"))
    put(["min_run", "min_bear_run"],                    vals.get("min_bear_run"))
    put(["min_bull_run"],                               vals.get("min_bull_run"))
    put(["mom_threshold", "mom_thr"],                   vals.get("mom_threshold"))
    put(["ddown_threshold", "dd_thr", "drawdown_threshold"],
                                                     vals.get("ddown_threshold"))
    put(["confirm_days", "confirm_bear"],               vals.get("confirm_days"))
    put(["bull_mom_threshold", "bull_mom_thr"],         vals.get("bull_mom_threshold"))
    put(["bull_ddown_exit", "bull_dd_exit"],            vals.get("bull_ddown_exit"))
    put(["confirm_days_bull", "confirm_bull"],          vals.get("confirm_days_bull"))
    put(["direction_gate"],                             vals.get("direction_gate"))
    put(["trend_gate"],                                 vals.get("trend_gate"))
    put(["entry_ret_lookback", "lbk", "lookback"],      vals.get("entry_ret_lookback"))
    put(["entry_ret_thresh", "entry_ret_thr"],          vals.get("entry_ret_thresh"))
    put(["entry_ddown_thresh", "entry_dd_thr"],         vals.get("entry_ddown_thresh"))
    put(["bear_profit_exit", "profit_exit"],            vals.get("bear_profit_exit"))
    put(["strict"],                                     vals.get("strict"))

    res = func(**out)
    # Normalize return
    if isinstance(res, tuple):
        if len(res) >= 2:
            return res[0], res[1]
        return res[0], None
    return res, None


# --- alias wrapper (tolerant to different parameter names) ---
_ALIAS_MAP = {
    "bear_enter":         ["prob_threshold", "enter_prob", "enter_threshold"],
    "bear_exit":          ["prob_exit", "exit_prob", "exit_threshold"],
    "k_forward":          ["k_fwd", "k"],
    "ema_span":           ["ema", "ema_smooth"],
    "min_bear_run":       ["min_run"],
    "ddown_threshold":    ["dd_thr", "drawdown_threshold"],
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
            if v is None:
                continue
            if k in sig_names:
                out[k] = v
            else:
                for alt in _ALIAS_MAP.get(k, ()):
                    if alt in sig_names:
                        out[alt] = v
                        break
        return out

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            return func(*args, **kwargs)
        return func(**_map_kwargs(kwargs))

    return wrapper

detect_regimes = _wrap_detect_regimes(detect_regimes)

df, model = _call_detect_regimes_flexible(
    detect_regimes,
    ticker=ticker,
    start="2000-01-01",
    end="today",
    n_components=n_components,
    k_forward=k_forward,
    ema_span=ema_span,
    bear_enter=bear_enter,
    bear_exit=bear_exit,
    min_bear_run=min_bear_run,
    min_bull_run=min_bull_run,
    mom_threshold=mom_threshold,
    ddown_threshold=ddown_threshold,
    confirm_days=confirm_days,
    bull_mom_threshold=bull_mom_threshold,
    bull_ddown_exit=bull_ddown_exit,
    confirm_days_bull=confirm_days_bull,
    direction_gate=direction_gate,
    trend_gate=trend_gate,
    entry_ret_lookback=entry_ret_lookback,
    entry_ret_thresh=entry_ret_thresh,
    entry_ddown_thresh=entry_ddown_thresh,
    bear_profit_exit=bear_profit_exit,
    strict=strict,
)

# df is assumed to have columns: 'Close', optional 'ema20'/'ema100',
# 'bear_candidate' (bool/int), 'bear_confirm' (bool/int) and optionally bull masks.
# If your function keeps EMAs separate, compute them here:
if "ema20" not in df.columns:
    df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
if "ema100" not in df.columns:
    df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()

# Last N years slice for regime plot
cutoff = df.index.max() - pd.DateOffset(years=zoom_years)
px_zoom = df.loc[df.index >= cutoff].copy()

def _safe_col(dframe: pd.DataFrame, name: str) -> pd.Series:
    """Strictly-boolean, aligned getter; returns all-False if column missing."""
    s = dframe.get(name, pd.Series(False, index=dframe.index))
    return pd.Series(s, index=dframe.index).fillna(False).astype(bool)

# -------- four regime masks (aligned to px_zoom) --------
# If bull masks aren't provided by the pipeline, infer them as complements of bear masks.
bear_cand = _safe_col(px_zoom, "bear_candidate")
bear_conf = _safe_col(px_zoom, "bear_confirm")

# optional bull masks; if missing, derive:
bull_cand_raw = px_zoom.get("bull_candidate")
bull_conf_raw = px_zoom.get("bull_confirm")

if bull_cand_raw is None or bull_conf_raw is None:
    # Define bull as NOT in bear (split candidate/confirmed by bear flags)
    bull_conf = (~bear_conf) & (px_zoom["Close"].notna())
    bull_cand = (~bear_cand) & (~bear_conf) & (px_zoom["Close"].notna())
else:
    bull_cand = _safe_col(px_zoom, "bull_candidate")
    bull_conf = _safe_col(px_zoom, "bull_confirm")

# candidate-only versions so confirmed doesn’t double-shade
bull_cand_only = bull_cand & (~bull_conf)
bear_cand_only = bear_cand & (~bear_conf)

# ======================= Save confirmed segments CSV (last 3y) =======================
from pathlib import Path as _Path

def _segments_from_mask(index, mask_bool, min_len=1):
    out, start = [], None
    arr = mask_bool.values if hasattr(mask_bool, "values") else mask_bool
    for i, v in enumerate(arr):
        v = bool(v)
        if v and start is None:
            start = index[i]
        if ((not v) or i == len(arr)-1) and start is not None:
            end = index[i] if (v and i == len(arr)-1) else index[i-1]
            if (end - start).days + 1 >= min_len:
                out.append((start, end))
            start = None
    return out

def _pick(val_series, ts, default=np.nan):
    try:
        return float(val_series.loc[ts])
    except Exception:
        return default

def _collect_confirmed_segments(px_view: pd.DataFrame, label: str, mask: pd.Series):
    rows = []
    mask = pd.Series(mask, index=px_view.index).astype(bool)
    for s, e in _segments_from_mask(px_view.index, mask, min_len=1):
        entry_price = _pick(px_view["Close"], s)
        exit_price  = _pick(px_view["Close"], e)
        days = (e - s).days + 1
        ret  = (exit_price / entry_price - 1.0) if (entry_price and exit_price and entry_price != 0) else np.nan
        rows.append({
            "type": label,
            "start": s.strftime("%Y-%m-%d"),
            "end":   e.strftime("%Y-%m-%d"),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "days": days,
            "return": ret,
        })
    return rows

rows = []
rows += _collect_confirmed_segments(px_zoom, "bull_confirm", bull_conf)
rows += _collect_confirmed_segments(px_zoom, "bear_confirm", bear_conf)

seg_df = pd.DataFrame(rows, columns=["type","start","end","entry_price","exit_price","days","return"])
out_dir = _Path(REPO_ROOT, "reports")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "confirmed_segments_last3y.csv"
seg_df.to_csv(out_csv, index=False)
st.caption(f"Saved confirmed segments (last {zoom_years}y) → `{out_csv}`")

# ======================= PLOTLY: TSLA regimes (last 3y) =======================
def _segments(index, mask_bool):
    """Return [(start_ts, end_ts), ...] of contiguous True runs."""
    out, start = [], None
    arr = mask_bool.values if hasattr(mask_bool, "values") else mask_bool
    for i in range(len(index)):
        m = bool(arr[i])
        if m and start is None:
            start = index[i]
        elif (not m) and (start is not None):
            out.append((start, index[i - 1]))
            start = None
    if start is not None:
        out.append((start, index[-1]))
    return out

def _add_band(fig, index, mask_bool, color, opacity, y0, y1):
    """Add background rectangles for True-runs in mask_bool over [y0,y1]."""
    for s, e in _segments(index, mask_bool):
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor=color, opacity=opacity,
            line_width=0, layer="below", y0=y0, y1=y1
        )

ymin = float(px_zoom["Close"].min()) * 0.95
ymax = float(px_zoom["Close"].max()) * 1.05

fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(
    x=px_zoom.index, y=px_zoom["Close"], name="Close",
    mode="lines", line=dict(width=2.0, color="#111")
))
fig_reg.add_trace(go.Scatter(
    x=px_zoom.index, y=px_zoom["ema20"], name="EMA20",
    mode="lines", line=dict(width=1.7, color="#ff7f0e")
))
fig_reg.add_trace(go.Scatter(
    x=px_zoom.index, y=px_zoom["ema100"], name="EMA100",
    mode="lines", line=dict(width=1.7, color="#2ca02c")
))

# --- 4-color shading (after traces, before layout) ---
_add_band(fig_reg, px_zoom.index, bull_cand_only, "#2ca02c", 0.12, ymin, ymax)  # light green  (candidate bull)
_add_band(fig_reg, px_zoom.index, bull_conf,      "#2ca02c", 0.30, ymin, ymax)  # dark  green  (confirmed bull)
_add_band(fig_reg, px_zoom.index, bear_cand_only, "#d62728", 0.12, ymin, ymax)  # light red    (candidate bear)
_add_band(fig_reg, px_zoom.index, bear_conf,      "#d62728", 0.30, ymin, ymax)  # dark  red    (confirmed bear)

# Title string mirrors your Colab chart
params_str = (
    f"k_fwd={k_forward}, EMA={ema_span}, enter={bear_enter:.2f}, exit={bear_exit:.2f}, "
    f"min_bear={min_bear_run}, min_bull={min_bull_run}, mom_thr={mom_threshold:.2f}, "
    f"dd_thr={ddown_threshold:.2f}, bull_mom_thr={bull_mom_threshold:.2f}, "
    f"bull_dd_exit={bull_ddown_exit:.2f}, confirm_bear={confirm_days}, "
    f"confirm_bull={confirm_days_bull}, dir_gate={direction_gate}, lbk={entry_ret_lookback}, "
    f"entry_ret_thr={entry_ret_thresh:.2f}, entry_dd_thr={entry_ddown_thresh:.2f}, "
    f"trend_gate={trend_gate}, profit_exit={bear_profit_exit:.2f}, strict={strict}"
)

fig_reg.update_layout(
    title=dict(text=f"TSLA — Regimes (last {zoom_years} years; light=candidate, dark=confirmed)<br>"
                    f"<sup>{params_str}</sup>",
               pad=dict(b=26)),
    template="plotly_white",
    height=520,
    margin=dict(l=10, r=10, t=90, b=10),
    legend=dict(
        orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
        bgcolor="rgba(255,255,255,0.85)"
    ),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", rangeslider=dict(visible=False)),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", range=[ymin, ymax]),
    hovermode="x unified",
)

st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")

# Optional PNG download (uses kaleido if present on the host)
try:
    st.download_button(
        "Download regimes chart (high-res PNG)",
        data=fig_reg.to_image(format="png", scale=3),  # scale>1 => higher DPI
        file_name=f"{ticker}_regimes_last{zoom_years}y.png",
        mime="image/png",
    )
except Exception:
    st.info("PNG export needs `kaleido`. Use the chart camera icon if PNG download is unavailable.")

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
