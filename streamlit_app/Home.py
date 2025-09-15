# streamlit_app/Home.py
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PURPOSE                                                                ║
# ║  -------                                                                ║
# ║  Streamlit page for TSLA showing:                                       ║
# ║    (1) Full-history Close + EMA20/EMA100                                ║
# ║    (2) Last-3-years Close + EMA20/EMA100                                ║
# ║    (3) Regime view with shading: green=bull, red=bear;                  ║
# ║        light=candidate, dark=confirmed                                   ║
# ║                                                                          ║
# ║  Notes                                                                   ║
# ║  • Fixed, hidden params (per your demo constraints):                     ║
# ║      n_components=4, zoom_years=3, entry_ret_lookback=10,                ║
# ║      entry_ret_thresh=-0.01                                              ║
# ║  • Two bull-quality knobs are visible again so you can shape runs:       ║
# ║      confirm_days_bull, min_bull_run                                     ║
# ║  • The code is resilient to small API changes in detect_regimes via      ║
# ║    a flexible adapter and sensible fallbacks if masks aren’t provided.   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ===== stdlib =====
import inspect
from pathlib import Path
import sys

# ===== third-party =====
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
# Repo import path: make `src/` importable when app runs from streamlit_app/
# ────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ────────────────────────────────────────────────────────────────────────────
# Import pipeline with friendly failure message
# ────────────────────────────────────────────────────────────────────────────
try:
    from src.regime_detection import detect_regimes
except Exception as e:
    st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
    st.error(
        "Could not import `src.regime_detection.detect_regimes`.\n"
        f"Repo root: {REPO_ROOT}\nError: {e}"
    )
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# Streamlit page configuration
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Regime Detection — HMM + Rules",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sidebar controls                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.title("TSLA Regime Detection — HMM + Configurable parameters")

ticker = "TSLA"

# Fixed (hidden) demo constraints
N_COMPONENTS_FIXED      = 4
ZOOM_YEARS_FIXED        = 3
ENTRY_RET_LOOKBACK_FIXED = 10
ENTRY_RET_THRESH_FIXED   = -0.01

# Visible knobs (kept minimal)
st.sidebar.header("Core configuration")
k_forward           = st.sidebar.slider("k_forward (label look-ahead days)", 1, 20, value=10, step=1)
ema_span            = st.sidebar.slider("EMA smoothing of bear prob", 5, 60, value=20, step=1)
bear_enter          = st.sidebar.slider("Bear enter threshold", 0.50, 0.99, value=0.72, step=0.01)
bear_exit           = st.sidebar.slider("Bear exit threshold", 0.05, 0.95, value=0.55, step=0.01)

st.sidebar.subheader("Confirmers (bear)")
mom_threshold       = st.sidebar.slider("Weakness threshold (EMA20<EMA100 by)", 0.00, 0.10, value=0.03, step=0.001)
ddown_threshold     = st.sidebar.slider("Drawdown threshold", 0.00, 0.30, value=0.15, step=0.005)
confirm_days        = st.sidebar.slider("Confirm days (bear)", 0, 20, value=7, step=1)
min_bear_run        = st.sidebar.slider("Min bear run (days)", 1, 60, value=15, step=1)

st.sidebar.subheader("Confirmers (bull)")
# ← You asked to expose these two again so you can shape bull runs.
confirm_days_bull   = st.sidebar.slider("Confirm days (bull)", 0, 10, value=3, step=1)
min_bull_run        = st.sidebar.slider("Min bull run (days)", 1, 60, value=5, step=1)

st.sidebar.subheader("Gates / exits")
direction_gate      = st.sidebar.checkbox("Directional gate (entry filter)", value=True)
trend_gate          = st.sidebar.checkbox("Trend gate (entry/exit with EMAs)", value=True)
bull_mom_threshold  = st.sidebar.slider("Bull momentum (EMA20>EMA100 by)", 0.00, 0.05, value=0.01, step=0.001)
bull_ddown_exit     = st.sidebar.slider("Bull drawdown exit (healed within)", 0.00, 0.20, value=0.06, step=0.005)
bear_profit_exit    = st.sidebar.slider("Bear profit-exit (bounce from entry)", 0.00, 0.20, value=0.05, step=0.005)
strict              = st.sidebar.checkbox("Strict direction cleanup (bear)", value=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Helpers                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _call_detect_regimes_flexible(func, **vals):
    """
    Map page knob names to whatever `detect_regimes` accepts (handles aliases).
    Always normalize return to (df, model).
    """
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    out = {}

    def put(names, value):
        for name in names:
            if name in params:
                out[name] = value
                return

    put(["ticker"],                 vals.get("ticker"))
    put(["start"],                  vals.get("start"))
    put(["end"],                    vals.get("end"))
    put(["n_components"],           vals.get("n_components"))
    put(["k_forward","k_fwd"],      vals.get("k_forward"))
    put(["ema_span","ema"],         vals.get("ema_span"))
    put(["prob_threshold","bear_enter","enter_prob","enter_threshold"], vals.get("bear_enter"))
    put(["prob_exit","bear_exit","exit_prob","exit_threshold"],        vals.get("bear_exit"))
    put(["min_run","min_bear_run"], vals.get("min_bear_run"))
    put(["min_bull_run"],           vals.get("min_bull_run"))
    put(["mom_threshold","mom_thr"], vals.get("mom_threshold"))
    put(["ddown_threshold","dd_thr","drawdown_threshold"], vals.get("ddown_threshold"))
    put(["confirm_days","confirm_bear"], vals.get("confirm_days"))
    put(["bull_mom_threshold","bull_mom_thr"], vals.get("bull_mom_threshold"))
    put(["bull_ddown_exit","bull_dd_exit"], vals.get("bull_ddown_exit"))
    put(["confirm_days_bull","confirm_bull"], vals.get("confirm_days_bull"))
    put(["direction_gate"],         vals.get("direction_gate"))
    put(["trend_gate"],             vals.get("trend_gate"))
    put(["entry_ret_lookback","lbk","lookback"], vals.get("entry_ret_lookback"))
    put(["entry_ret_thresh","entry_ret_thr"],    vals.get("entry_ret_thresh"))
    put(["entry_ddown_thresh","entry_dd_thr"],   vals.get("entry_ddown_thresh"))
    put(["bear_profit_exit","profit_exit"],      vals.get("bear_profit_exit"))
    put(["strict","strict_direction"],           vals.get("strict"))

    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)

def _ensure_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee EMA columns exist for plotting / confirmation rules."""
    if df is None or df.empty:
        return df
    if "Close" in df:
        if "ema20" not in df.columns:
            df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "ema100" not in df.columns:
            df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df

def _last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty: return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()

def _segments(index, mask_bool):
    """Convert boolean mask to contiguous (start, end) spans."""
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

def _add_vbands(fig, idx, mask_bool, color, opacity):
    """
    Shade vertical bands spanning full plot height (yref='paper').
    Works reliably regardless of y-axis range.
    """
    for s, e in _segments(idx, mask_bool):
        fig.add_shape(
            type="rect",
            x0=s, x1=e, xref="x",
            y0=0, y1=1, yref="paper",
            fillcolor=color, opacity=opacity,
            line=dict(width=0), layer="below",
        )

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Run pipeline once and build views                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with st.spinner("Running regime pipeline (TSLA)…"):
    df, _ = _call_detect_regimes_flexible(
        detect_regimes,
        ticker=ticker, start="2000-01-01", end="today",
        n_components=N_COMPONENTS_FIXED,           # fixed (hidden)
        k_forward=k_forward,
        ema_span=ema_span,
        bear_enter=bear_enter, bear_exit=bear_exit,
        min_bear_run=min_bear_run, min_bull_run=min_bull_run,  # min_bull_run now visible
        mom_threshold=mom_threshold, ddown_threshold=ddown_threshold,
        confirm_days=confirm_days,
        bull_mom_threshold=bull_mom_threshold, bull_ddown_exit=bull_ddown_exit,
        confirm_days_bull=confirm_days_bull,       # visible again
        direction_gate=direction_gate, trend_gate=trend_gate,
        entry_ret_lookback=ENTRY_RET_LOOKBACK_FIXED,   # fixed (hidden)
        entry_ret_thresh=ENTRY_RET_THRESH_FIXED,       # fixed (hidden)
        entry_ddown_thresh=-0.03,                      # keep your earlier default
        bear_profit_exit=bear_profit_exit,
        strict=strict,
    )

if df is None or df.empty or "Close" not in df.columns:
    st.error("Pipeline returned no data or missing 'Close'.")
    st.stop()

df = _ensure_emas(df.sort_index())


# --- Full-history Close strictly from Yahoo ---
import yfinance as yf

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_close_full(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_localize(None)
    out = raw[["Close"]].dropna().copy()
    # EMAs 
    out["ema20"]  = out["Close"].ewm(span=20,  adjust=False).mean()
    out["ema100"] = out["Close"].ewm(span=100, adjust=False).mean()
    return out

px_full = _fetch_close_full(ticker)             # <-- true IPO→today
px_zoom = px_full.loc[px_full.index >= px_full.index.max() - pd.DateOffset(years=ZOOM_YEARS_FIXED)].copy()

def _plot_close_emas(dfp: pd.DataFrame, title: str, h=440) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfp.index, y=dfp["Close"],  name="Close",
                             mode="lines", line=dict(width=2.1, color="#111")))
    fig.add_trace(go.Scatter(x=dfp.index, y=dfp["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.5, color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=dfp.index, y=dfp["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.5, color="#2ca02c")))
    fig.update_layout(
        title=dict(text=title, pad=dict(b=22)),
        template="plotly_white", height=h,
        margin=dict(l=10, r=10, t=72, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig

# (1) IPO→today
st.plotly_chart(_plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)"),
                use_container_width=True, theme="streamlit")
# (2) last 3 years
st.plotly_chart(_plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {ZOOM_YEARS_FIXED} years)"),
                use_container_width=True, theme="streamlit")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Regime masks (prefer pipeline-provided; infer if missing/empty)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _infer_masks_for(df_all: pd.DataFrame):
    """Return (bull_cand, bull_conf, bear_cand, bear_conf) aligned to df_all.index."""
    idx = df_all.index

    # Prefer pipeline columns if present
    bc = pd.Series(df_all.get("bear_candidate", False), index=idx).astype(bool)
    bf = pd.Series(df_all.get("bear_confirm",   False), index=idx).astype(bool)
    ul = pd.Series(df_all.get("bull_candidate", False), index=idx).astype(bool)
    uf = pd.Series(df_all.get("bull_confirm",   False), index=idx).astype(bool)

    have_pipe = (ul.any() or uf.any() or bc.any() or bf.any())

    if not have_pipe:
        # Fallback: infer from p_bear (or compute a proxy) + EMA relation + drawdown
        def _prob_like(d):
            # pick a column that looks like bear probability
            for c in d.columns:
                cl = c.lower()
                if "p_bear" in cl or ("prob" in cl and "bear" in cl):
                    return pd.to_numeric(d[c], errors="coerce")
            return pd.Series(index=d.index, dtype=float)

        p_bear = _prob_like(df_all).fillna(0.0)
        p_bear_ema = p_bear.ewm(span=ema_span, adjust=False).mean()

        bear_cand = (p_bear_ema >= bear_enter)
        ema_ok_bear = df_all["ema20"] < df_all["ema100"] * (1 - mom_threshold)
        dd = (df_all["Close"] / df_all["Close"].cummax() - 1.0)
        dd_ok_bear = dd <= -ddown_threshold
        bear_conf = bear_cand & ema_ok_bear & dd_ok_bear

        bull_cand = ~bear_cand  # simple complement as a fallback
        ema_ok_bull = df_all["ema20"] > df_all["ema100"] * (1 + bull_mom_threshold)
        # bull confirm when not in deep drawdown anymore (healed)
        dd_ok_bull = dd >= -bull_ddown_exit
        bull_conf = bull_cand & ema_ok_bull & dd_ok_bull
    else:
        bull_cand, bull_conf, bear_cand, bear_conf = ul, uf, bc, bf

    # Apply min-run hygiene
    def _prune(mask: pd.Series, min_len: int) -> pd.Series:
        mask = mask.astype(bool).copy()
        run_id = (mask != mask.shift()).cumsum()
        kept = mask.copy()
        for _, seg in mask.groupby(run_id):
            if seg.iloc[0] and len(seg) < min_len:
                kept.loc[seg.index] = False
        return kept

    bear_cand = _prune(bear_cand, min_bear_run)
    bear_conf = _prune(bear_conf, max(1, min_bear_run // 2))
    bull_cand = _prune(bull_cand, min_bull_run)
    bull_conf = _prune(bull_conf, max(1, min_bull_run // 2))

    return bull_cand.astype(bool), bull_conf.astype(bool), bear_cand.astype(bool), bear_conf.astype(bool)

# Build masks and reindex to the zoom window
bull_cand_all, bull_conf_all, bear_cand_all, bear_conf_all = _infer_masks_for(df)
bull_cand = bull_cand_all.reindex(px_zoom.index).fillna(False)
bull_conf = bull_conf_all.reindex(px_zoom.index).fillna(False)
bear_cand = bear_cand_all.reindex(px_zoom.index).fillna(False)
bear_conf = bear_conf_all.reindex(px_zoom.index).fillna(False)

# Candidate-only masks for visual clarity (don’t double shade where confirmed)
bull_cand_only = bull_cand & (~bull_conf)
bear_cand_only = bear_cand & (~bear_conf)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Regime chart (last 3 years)                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["Close"],  name="Close",
                             mode="lines", line=dict(width=2.0, color="#111")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.5, color="#ff7f0e")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.5, color="#2ca02c")))

# Shade: green=bull, red=bear (light=candidate, dark=confirmed)
_add_vbands(fig_reg, px_zoom.index, bull_cand_only, "rgba(46, 204, 113, 0.35)", 0.35)
_add_vbands(fig_reg, px_zoom.index, bull_conf,      "rgba(39, 174, 96, 0.55)",  0.55)
_add_vbands(fig_reg, px_zoom.index, bear_cand_only, "rgba(231, 76, 60, 0.35)",  0.35)
_add_vbands(fig_reg, px_zoom.index, bear_conf,      "rgba(192, 57, 43, 0.55)",  0.55)

params_str = (
    f"k_fwd={k_forward}, EMA={ema_span}, enter={bear_enter:.2f}, exit={bear_exit:.2f}, "
    f"min_bear={min_bear_run}, min_bull={min_bull_run}, mom_thr={mom_threshold:.2f}, "
    f"dd_thr={ddown_threshold:.2f}, bull_mom_thr={bull_mom_threshold:.2f}, "
    f"bull_dd_exit={bull_ddown_exit:.2f}, confirm_bear={confirm_days}, "
    f"confirm_bull={confirm_days_bull}, dir_gate={direction_gate}, "
    f"bear_profit_exit={bear_profit_exit:.2f}, strict={strict}"
)
fig_reg.update_layout(
    title=dict(text=f"TSLA — Regimes (last {ZOOM_YEARS_FIXED} years; green=bull, red=bear; light=candidate, dark=confirmed)<br>"
                    f"<sup>{params_str}</sup>", pad=dict(b=26)),
    template="plotly_white", height=540,
    margin=dict(l=10, r=10, t=96, b=10),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    hovermode="x unified",
)
st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Confirmed segments tables (copyable)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _seg_rows(mask: pd.Series, label: str):
    rows = []
    for s, e in _segments(mask.index, mask.values):
        start_close = float(px_zoom.loc[s, "Close"]) if s in px_zoom.index else None
        end_close   = float(px_zoom.loc[e, "Close"]) if e in px_zoom.index else None
        ret = (end_close / start_close - 1.0) if (start_close and end_close and start_close != 0) else None
        rows.append({"type": label, "start": s, "end": e,
                     "days": (e - s).days + 1, "start_close": start_close,
                     "end_close": end_close})
    return rows

bull_rows = _seg_rows(bull_conf, "bull_confirm")
bear_rows = _seg_rows(bear_conf, "bear_confirm")

st.subheader("Confirmed segments in zoom window")
cols = st.columns(2, gap="large")

with cols[0]:
    st.markdown("**Bull — confirmed**")
    st.dataframe(pd.DataFrame(bull_rows), use_container_width=True)

with cols[1]:
    st.markdown("**Bear — confirmed**")
    st.dataframe(pd.DataFrame(bear_rows), use_container_width=True)

st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
