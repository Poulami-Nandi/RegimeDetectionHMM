# streamlit_app/Home.py
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PURPOSE                                                                ║
# ║  -------                                                                ║
# ║  This Streamlit page renders three charts for TSLA:                     ║
# ║    1) IPO→today Close with EMA20/EMA100                                  ║
# ║    2) Last N years Close with EMA20/EMA100 (zoom view)                   ║
# ║    3) Last N years "Regime" view with light (candidate) and dark         ║
# ║       (confirmed) bear shading.                                          ║
# ║                                                                          ║
# ║  The page is intentionally resilient to small API differences in your   ║
# ║  pipeline: it calls `detect_regimes(...)` via a flexible adapter and     ║
# ║  falls back to simple inference for bear candidate/confirm masks if the  ║
# ║  pipeline does not provide them directly.                                ║
# ║                                                                          ║
# ║  Note: This page is fixed to TSLA on purpose to make demo behavior       ║
# ║  deterministic.                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ===== stdlib =====
import inspect
from functools import wraps
from pathlib import Path
import sys

# ===== third-party =====
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
# Import path setup
# - Streamlit Cloud often runs the app from /mount/... and repo code lives
#   one level up from `streamlit_app/`. We add that repo root to sys.path
#   so `src.regime_detection` becomes importable.
# ────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ────────────────────────────────────────────────────────────────────────────
# Pipeline import (with friendly failure)
# If this import fails, we surface a helpful error that shows the repo root
# Streamlit detected (useful when debugging deployment path issues).
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

# ===========================================================================#
#                                SIDEBAR                                     #
# ===========================================================================#
# The app is locked to TSLA so other parts of the repo (e.g. a separate
# page for model comparisons) can assume consistent data.
st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")
st.sidebar.markdown("### Controls (fixed to TSLA)")
ticker = "TSLA"

# All knobs are surfaced here to mirror your Colab defaults and allow
# quick experimentation. The detect_regimes adapter below tries multiple
# parameter name aliases so your internal function can evolve without
# breaking this UI.
st.sidebar.markdown("### Regime knobs (defaults ~ Colab)")
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

# ===========================================================================#
#                              HELPERS (UI-safe)                             #
# ===========================================================================#

def _call_detect_regimes_flexible(func, **vals):
    """
    Adapter that maps the page's knob names to whatever the current
    `detect_regimes` signature expects. This protects the page from
    minor API changes (e.g., renamed args) by trying several aliases.

    Returns
    -------
    (df, model) : tuple
        We normalize to `(df, model)` even if the underlying function
        returns only a DataFrame (in which case model=None).
    """
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    out = {}

    def put(names, value):
        # Assign value to the first matching parameter name in `names`
        for name in names:
            if name in params:
                out[name] = value
                return

    # Core / common names + aliases used in earlier iterations of the pipeline
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
    put(["strict"],                 vals.get("strict"))

    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)


def _ensure_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee EMA columns exist (ema20, ema100) even if the pipeline
    does not return them. Keeps the plotting code simple and robust.
    """
    if df is None or df.empty:
        return df
    if "Close" in df:
        if "ema20" not in df.columns:
            df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "ema100" not in df.columns:
            df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df


def _last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """
    Slice the *tail* of a DataFrame to the last `years` worth of rows
    using the DateTimeIndex. Returns a copy.
    """
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()


def _try_get_prob_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Best-effort search for a “bear probability” series if the pipeline
    exposes one under a non-standard name. We look for any column that
    contains "prob" and "bear" (case-insensitive) and cast to float.
    """
    candidates = [c for c in df.columns if "prob" in c.lower() and "bear" in c.lower()]
    if candidates:
        s = pd.to_numeric(df[candidates[0]], errors="coerce")
        return s.astype(float)
    return None


def _infer_bear_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Construct (bear_candidate, bear_confirm) as boolean Series aligned to
    df.index, **only if** the pipeline didn’t provide them.

    Candidate:
      - If `bear_candidate` exists, use it directly.
      - Else, use an EMA-smoothed bear probability and compare to `bear_enter`.

    Confirmed:
      - Candidate & (EMA20 below EMA100 by `mom_threshold`)
                 & (drawdown ≤ -`ddown_threshold`)
      - We apply a short run-length filter so isolated single-day spikes
        don’t appear as confirmed.

    Returns
    -------
    (cand, conf): tuple[pd.Series, pd.Series]
    """
    idx = df.index
    cand = pd.Series(False, index=idx)

    # Prefer explicitly provided mask from the pipeline
    if "bear_candidate" in df:
        cand = pd.Series(df["bear_candidate"], index=idx).fillna(False).astype(bool)
    else:
        p = _try_get_prob_series(df)
        if p is not None:
            # hysteresis *entry* proxy when no explicit candidate mask is given
            cand = (p.ewm(span=ema_span, adjust=False).mean() >= bear_enter).reindex(idx).fillna(False)

    # Ensure EMAs exist for confirmation rules
    tmp = _ensure_emas(df.copy())

    # Trend: EMA20 under EMA100 by at least mom_threshold (as a fraction)
    ema_ok = (tmp["ema20"] < tmp["ema100"] * (1 - mom_threshold)).reindex(idx).fillna(False)

    # Drawdown: distance from rolling peak must exceed ddown_threshold
    rolling_peak = tmp["Close"].cummax()
    dd = (tmp["Close"] / rolling_peak - 1.0).reindex(idx)
    dd_ok = (dd <= -ddown_threshold).fillna(False)

    conf = (cand & ema_ok & dd_ok)

    # Lightweight run-length pruning to keep charts clean
    def _min_run_filter(mask: pd.Series, min_len: int) -> pd.Series:
        mask = mask.astype(bool).copy()
        run_id = (mask != mask.shift()).cumsum()
        kept = mask.copy()
        for _, seg in mask.groupby(run_id):
            if seg.iloc[0] and len(seg) < min_len:
                kept.loc[seg.index] = False
        return kept

    cand = _min_run_filter(cand, min_bear_run)
    conf = _min_run_filter(conf, max(1, min_bear_run // 2))
    return cand.astype(bool), conf.astype(bool)


def _segments(index, mask_bool):
    """
    Utility for shading: convert a boolean mask into contiguous
    (start, end) tuples in index space.
    """
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
    """
    Add vertical background rectangles to `fig` anywhere `mask_bool` is True.
    We paint them on the 'below' layer so the price lines stay visible.
    """
    for s, e in _segments(idx, mask_bool):
        fig.add_vrect(
            x0=s, x1=e, fillcolor=color, opacity=opacity,
            line_width=0, layer="below", y0=y0, y1=y1
        )

# ===========================================================================#
#                     RUN PIPELINE (ONCE) AND PLOT                           #
# ===========================================================================#
with st.spinner("Running regime pipeline (TSLA)…"):
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

# Defensive checks: if the pipeline returns an empty DF or without 'Close',
# there is nothing to plot.
if df is None or df.empty or "Close" not in df.columns:
    st.error("Pipeline returned no data or missing 'Close'.")
    st.stop()

# Sort, backfill EMAs (if missing) and create a zoom slice for the second/third charts
df = df.sort_index()
df = _ensure_emas(df)
px_full = df[["Close","ema20","ema100"]].copy()
px_zoom = _last_years(px_full, zoom_years)

# ────────────────────────────────────────────────────────────────────────────
# Basic price+EMA plotting helper
# ────────────────────────────────────────────────────────────────────────────
def _plot_close_emas(df_plot: pd.DataFrame, title: str, h=440) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Close",
                             mode="lines", line=dict(width=2.2, color="#111")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.6, color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.6, color="#2ca02c")))
    fig.update_layout(
        title=dict(text=title, pad=dict(b=26)),
        template="plotly_white",
        height=h,
        margin=dict(l=10, r=10, t=78, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig

# 1) Full history (IPO→today) — uses pipeline output to avoid a second Yahoo call
fig1 = _plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)")
st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

# 2) Last-N-years zoom slice
fig2 = _plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)")
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

# ────────────────────────────────────────────────────────────────────────────
# 3) Regime view (zoom window) with shading
#    - We prefer pipeline-provided masks: 'bear_candidate' + 'bear_confirm'
#    - Else we infer from probability + trend/drawdown rules
# ────────────────────────────────────────────────────────────────────────────
# --------- Regime view with guaranteed masks + rich debug ----------

# 1) narrow to the zoom window using your pipeline output (guarantees EMAs exist earlier)
px_full = df[["Close", "ema20", "ema100"]].copy()
px_zoom = _last_years(px_full, zoom_years)

# Helper: contiguous True segments
def _segments(index, mask_bool):
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

# Helper: small-run cleaner
def _min_run_filter(mask: pd.Series, min_len: int) -> pd.Series:
    mask = pd.Series(mask, index=mask.index, dtype=bool)
    run_id = (mask != mask.shift()).cumsum()
    kept = mask.copy()
    for _, seg in mask.groupby(run_id):
        if seg.iloc[0] and len(seg) < min_len:
            kept.loc[seg.index] = False
    return kept

# Helper: robustly infer candidate/confirm if they’re missing/empty
def _infer_bear_masks(df_all: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = df_all.index

    # try to locate a "bear probability" column (common names)
    prob_cols = [c for c in df_all.columns if "bear" in c.lower() and "prob" in c.lower()]
    if prob_cols:
        p_bear = pd.to_numeric(df_all[prob_cols[0]], errors="coerce").astype(float)
    elif "p_bear" in df_all.columns:
        p_bear = pd.to_numeric(df_all["p_bear"], errors="coerce").astype(float)
    else:
        # last resort — if model exposed per-state posts, add your own mapping as needed
        p_bear = pd.Series(index=idx, dtype=float)

    p_bear_ema = p_bear.ewm(span=ema_span, adjust=False).mean()

    # Candidate: smoothed probability crosses enter threshold
    cand = (p_bear_ema >= bear_enter).reindex(idx).fillna(False)

    # Confirmation: require both trend + drawdown
    ema_ok = (df_all["ema20"] < df_all["ema100"] * (1 - mom_threshold)).reindex(idx).fillna(False)
    # drawdown (≤0). If your features include one, prefer that; else compute quick local version:
    if "drawdown" in df_all.columns:
        dd = pd.to_numeric(df_all["drawdown"], errors="coerce")
    else:
        dd = (df_all["Close"] / df_all["Close"].cummax() - 1.0)
    dd_ok = (dd <= -ddown_threshold).fillna(False)

    conf = (cand & ema_ok & dd_ok)

    # Run-length hygiene
    cand_f = _min_run_filter(cand, min_bear_run)
    conf_f = _min_run_filter(conf, max(1, min_bear_run // 2))

    return cand_f.astype(bool), conf_f.astype(bool), p_bear, p_bear_ema

# 2) Choose masks: use pipeline masks ONLY if they contain any True; otherwise use inferred
bear_cand_raw = pd.Series(df.get("bear_candidate", False), index=df.index).astype(bool)
bear_conf_raw = pd.Series(df.get("bear_confirm",   False), index=df.index).astype(bool)

# Build inferred masks from probability + EMA + drawdown rules
cand_inf, conf_inf, p_bear, p_bear_ema = _infer_bear_masks(df)

# NEW: prefer pipeline masks only when they actually have signals
bear_cand_used = bear_cand_raw if bear_cand_raw.any() else cand_inf
bear_conf_used = bear_conf_raw if bear_conf_raw.any() else conf_inf

# 3) Reindex masks to zoom window & build candidate-only view
bear_cand = bear_cand_used.reindex(px_zoom.index).fillna(False)
bear_conf = bear_conf_used.reindex(px_zoom.index).fillna(False)
cand_only = (bear_cand & (~bear_conf))

# 4) Third plot: TSLA + EMAs + shading
ymin = float(px_zoom["Close"].min()) * 0.95
ymax = float(px_zoom["Close"].max()) * 1.05

fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["Close"],  name="Close",
                             mode="lines", line=dict(width=2.0, color="#111")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.6, color="#ff7f0e")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.6, color="#2ca02c")))

def _add_vbands(fig, idx, mask_bool, color, opacity, y0, y1):
    for s, e in _segments(idx, mask_bool):
        fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=opacity,
                      line_width=0, layer="below", y0=y0, y1=y1)

# 4-color scheme readiness (bear now; bull not drawn in this file)
_add_vbands(fig_reg, px_zoom.index, cand_only, "crimson", 0.12, ymin, ymax)  # candidate bear (light red)
_add_vbands(fig_reg, px_zoom.index, bear_conf, "crimson", 0.30, ymin, ymax)  # confirmed bear (dark red)

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

# 5) Debug UI (toggle in sidebar)
show_debug = st.sidebar.checkbox("Debug mode (regimes)", value=False)

if show_debug:
    st.markdown("### Regime debug")
    with st.expander("Input/columns snapshot", expanded=True):
        st.write({
            "df.shape": df.shape,
            "index.range": f"{df.index.min()} → {df.index.max()}",
            "has_bear_candidate": bool("bear_candidate" in df.columns),
            "has_bear_confirm": bool("bear_confirm" in df.columns),
            "any(bear_candidate)": bool(bear_cand_raw.any()),
            "any(bear_confirm)":  bool(bear_conf_raw.any()),
            "used_pipeline_masks": bool(have_pipe_masks),
        })
        st.write("Columns:", list(df.columns)[:40])

    # Show counts on the zoom index (final masks used for shading)
    with st.expander("Mask counts on the zoom window", expanded=True):
        st.write({
            "pipeline_candidate.any": bool(bear_cand_raw.any()),
            "pipeline_confirm.any":  bool(bear_conf_raw.any()),
            "candidate_used.sum": int(bear_cand.sum()),
            "confirmed_used.sum": int(bear_conf.sum()),
            "candidate_inferred.sum": int(cand_inf.reindex(px_zoom.index).fillna(False).sum()),
            "confirmed_inferred.sum": int(conf_inf.reindex(px_zoom.index).fillna(False).sum()),
        })

    # p_bear line and thresholds — this explains “why nothing shades”
    if not p_bear.empty:
        pz = p_bear.reindex(px_zoom.index)
        pz_ema = p_bear_ema.reindex(px_zoom.index)
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=pz.index, y=pz.values, name="p_bear", mode="lines"))
        fig_prob.add_trace(go.Scatter(x=pz_ema.index, y=pz_ema.values, name=f"EMA({ema_span})", mode="lines"))
        # Enter / Exit levels (horizontal lines for context)
        if not pd.isna(bear_enter):
            fig_prob.add_hline(y=bear_enter, line=dict(width=1, dash="dash"), annotation_text="enter", annotation_position="top left")
        if not pd.isna(bear_exit):
            fig_prob.add_hline(y=bear_exit, line=dict(width=1, dash="dot"), annotation_text="exit", annotation_position="bottom left")
        fig_prob.update_layout(height=260, template="plotly_white", title="Bear probability (raw & EMA)")
        st.plotly_chart(fig_prob, use_container_width=True)

    # List start/end of shaded intervals for the plot you’re looking at
    def _seg_rows(mask: pd.Series, label: str):
        rows = []
        for s, e in _segments(mask.index, mask.values):
            rows.append({"type": label, "start": s, "end": e, "days": (e - s).days + 1})
        return rows

    seg_rows = _seg_rows(cand_only, "bear_candidate_only") + _seg_rows(bear_conf, "bear_confirm")
    seg_df = pd.DataFrame(seg_rows)
    if seg_df.empty:
        st.info("No segments found in the zoom window. If you expect bears, lower `bear_enter`, decrease `ema_span`, or reduce `min_bear_run`.")
    else:
        st.dataframe(seg_df, use_container_width=True)

# Footer reminder (compliance / demo clarity)
st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
