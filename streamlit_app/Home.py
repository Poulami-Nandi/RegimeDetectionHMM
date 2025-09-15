# streamlit_app/Home.py
from __future__ import annotations

# ===== stdlib =====
import inspect
from pathlib import Path
import sys

# ===== third-party =====
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── import path ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── pipeline import (friendly failure) ──────────────────────────────────────
try:
    from src.regime_detection import detect_regimes
except Exception as e:
    st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
    st.error(
        "Could not import `src.regime_detection.detect_regimes`.\n"
        f"Repo root: {REPO_ROOT}\nError: {e}"
    )
    st.stop()

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="TSLA Regime Detection — HMM + Rules",
                   layout="wide", initial_sidebar_state="expanded")

st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")
st.sidebar.markdown("### Controls (fixed to TSLA)")
ticker = "TSLA"

# ── knobs (same as before) ──────────────────────────────────────────────────
n_components        = st.sidebar.number_input("HMM states (n_components)", 2, 6, 4, 1)
k_forward           = st.sidebar.slider("k_forward (days ahead for label)", 1, 20, 10, 1)
ema_span            = st.sidebar.slider("EMA smoothing of bear prob (ema_span)", 5, 60, 20, 1)
bear_enter          = st.sidebar.slider("Bear enter threshold", 0.50, 0.99, 0.80, 0.01)
bear_exit           = st.sidebar.slider("Bear exit threshold", 0.00, 0.95, 0.55, 0.01)
min_bear_run        = st.sidebar.slider("Min bear run (days)", 1, 60, 15, 1)
min_bull_run        = st.sidebar.slider("Min bull run (days)", 1, 60, 5, 1)
mom_threshold       = st.sidebar.slider("Trend weakness (mom_threshold)", 0.00, 0.10, 0.03, 0.001)
ddown_threshold     = st.sidebar.slider("Drawdown confirm (ddown_threshold)", 0.00, 0.30, 0.15, 0.005)
confirm_days        = st.sidebar.slider("Confirm days (bear)", 0, 20, 7, 1)
bull_mom_threshold  = st.sidebar.slider("Bull trend (bull_mom_threshold)", 0.00, 0.05, 0.01, 0.001)
bull_ddown_exit     = st.sidebar.slider("Bull dd exit (bull_ddown_exit)", 0.00, 0.20, 0.06, 0.005)
confirm_days_bull   = st.sidebar.slider("Confirm days (bull)", 0, 10, 3, 1)
direction_gate      = st.sidebar.checkbox("Directional gate", True)
trend_gate          = st.sidebar.checkbox("Trend gate", True)
strict              = st.sidebar.checkbox("Strict confirmation", False)
entry_ret_lookback  = st.sidebar.slider("Entry return lookback (days)", 1, 30, 10, 1)
entry_ret_thresh    = st.sidebar.slider("Entry return threshold", -0.05, 0.05, -0.01, 0.001)
entry_ddown_thresh  = st.sidebar.slider("Entry drawdown threshold", -0.10, 0.10, -0.03, 0.001)
bear_profit_exit    = st.sidebar.slider("Bear profit exit", 0.00, 0.20, 0.05, 0.005)
zoom_years          = st.sidebar.slider("Zoom window (years)", 1, 10, 3, 1)

# ── helpers ─────────────────────────────────────────────────────────────────
def _call_detect_regimes_flexible(func, **vals):
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    out = {}
    def put(names, value):
        for n in names:
            if n in params:
                out[n] = value; return
    put(["ticker"], vals.get("ticker"))
    put(["start"], vals.get("start"))
    put(["end"], vals.get("end"))
    put(["n_components"], vals.get("n_components"))
    put(["k_forward","k_fwd"], vals.get("k_forward"))
    put(["ema_span","ema"], vals.get("ema_span"))
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
    put(["direction_gate"], vals.get("direction_gate"))
    put(["trend_gate"],     vals.get("trend_gate"))
    put(["entry_ret_lookback","lbk","lookback"], vals.get("entry_ret_lookback"))
    put(["entry_ret_thresh","entry_ret_thr"],    vals.get("entry_ret_thresh"))
    put(["entry_ddown_thresh","entry_dd_thr"],   vals.get("entry_ddown_thresh"))
    put(["bear_profit_exit","profit_exit"],      vals.get("bear_profit_exit"))
    put(["strict"],           vals.get("strict"))
    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)

def _ensure_emas(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Close" in df:
        if "ema20" not in df:  df["ema20"]  = df["Close"].ewm(span=20,  adjust=False).mean()
        if "ema100" not in df: df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df

def _last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty: return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()

def _try_get_prob_series(df: pd.DataFrame) -> pd.Series | None:
    cands = [c for c in df.columns if "prob" in c.lower() and "bear" in c.lower()]
    if cands:
        s = pd.to_numeric(df[cands[0]], errors="coerce")
        return s.astype(float)
    return None

def _min_run_filter(mask: pd.Series, min_len: int) -> pd.Series:
    mask = pd.Series(mask, index=mask.index, dtype=bool)
    run_id = (mask != mask.shift()).cumsum()
    kept = mask.copy()
    for _, seg in mask.groupby(run_id):
        if seg.iloc[0] and len(seg) < min_len:
            kept.loc[seg.index] = False
    return kept

def _infer_bear_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = df.index
    p_bear = _try_get_prob_series(df)
    if p_bear is None:
        p_bear = pd.Series(index=idx, dtype=float)
    p_bear_ema = p_bear.ewm(span=ema_span, adjust=False).mean()
    # candidate: smoothed prob >= enter
    cand = (p_bear_ema >= bear_enter).reindex(idx).fillna(False)
    # confirm: trend & drawdown
    tmp = _ensure_emas(df.copy())
    ema_ok = (tmp["ema20"] < tmp["ema100"] * (1 - mom_threshold)).reindex(idx).fillna(False)
    if "drawdown" in tmp:
        dd = pd.to_numeric(tmp["drawdown"], errors="coerce")
    else:
        dd = (tmp["Close"] / tmp["Close"].cummax() - 1.0)
    dd_ok = (dd <= -ddown_threshold).fillna(False)
    conf = cand & ema_ok & dd_ok
    return _min_run_filter(cand, min_bear_run), _min_run_filter(conf, max(1, min_bear_run//2)), p_bear_ema

def _infer_bull_masks(df: pd.DataFrame, p_bear_ema: pd.Series | None) -> tuple[pd.Series, pd.Series]:
    idx = df.index
    tmp = _ensure_emas(df.copy())
    # If we have p_bear_ema, use hysteresis complement for candidate; else use simple trend candidate
    if p_bear_ema is not None and not p_bear_ema.empty:
        bull_cand = (p_bear_ema <= bear_exit).reindex(idx).fillna(False)
    else:
        bull_cand = (tmp["ema20"] > tmp["ema100"]).reindex(idx).fillna(False)
    # Confirm bull: trend strong AND drawdown healed
    ema_up = (tmp["ema20"] > tmp["ema100"] * (1 + max(0.0001, bull_mom_threshold))).reindex(idx).fillna(False)
    if "drawdown" in tmp:
        dd = pd.to_numeric(tmp["drawdown"], errors="coerce")
    else:
        dd = (tmp["Close"] / tmp["Close"].cummax() - 1.0)
    dd_healed = (dd >= -bull_ddown_exit).fillna(False)
    bull_conf = bull_cand & ema_up & dd_healed
    return _min_run_filter(bull_cand, min_bull_run), _min_run_filter(bull_conf, max(1, min_bull_run//2))

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

def _add_vbands(fig, idx, mask_bool, color, opacity):
    for s, e in _segments(idx, mask_bool):
        fig.add_shape(
            type="rect", x0=s, x1=e,
            xref="x", y0=0, y1=1, yref="paper",
            fillcolor=color, opacity=opacity, line=dict(width=0), layer="below"
        )

# ── run pipeline ────────────────────────────────────────────────────────────
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

if df is None or df.empty or "Close" not in df.columns:
    st.error("Pipeline returned no data or missing 'Close'."); st.stop()

df = _ensure_emas(df.sort_index())
px_full = df[["Close","ema20","ema100"]].copy()
px_zoom = _last_years(px_full, zoom_years)

# ── price plots (unchanged) ────────────────────────────────────────────────
def _plot_close_emas(df_plot: pd.DataFrame, title: str, h=440) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Close",
                             mode="lines", line=dict(width=2.2, color="#111")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.6, color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.6, color="#2ca02c")))
    fig.update_layout(
        title=dict(text=title, pad=dict(b=26)), template="plotly_white", height=h,
        margin=dict(l=10, r=10, t=78, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    ); return fig

st.plotly_chart(_plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)"),
                use_container_width=True, theme="streamlit")
st.plotly_chart(_plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)"),
                use_container_width=True, theme="streamlit")

# ── regime view (now all 4 states) ─────────────────────────────────────────
# Get pipeline masks if present; otherwise infer. Also infer bull masks.
bear_cand_raw = pd.Series(df.get("bear_candidate", False), index=df.index).astype(bool)
bear_conf_raw = pd.Series(df.get("bear_confirm",   False), index=df.index).astype(bool)

cand_inf, conf_inf, p_bear_ema = _infer_bear_masks(df)
bear_cand_used = bear_cand_raw if bear_cand_raw.any() else cand_inf
bear_conf_used = bear_conf_raw if bear_conf_raw.any() else conf_inf

# Bulls (from pipeline if present, else inferred)
bull_cand_raw = pd.Series(df.get("bull_candidate", False), index=df.index).astype(bool)
bull_conf_raw = pd.Series(df.get("bull_confirm",   False), index=df.index).astype(bool)
bull_cand_inf, bull_conf_inf = _infer_bull_masks(df, p_bear_ema)

bull_cand_used = bull_cand_raw if bull_cand_raw.any() else bull_cand_inf
bull_conf_used = bull_conf_raw if bull_conf_raw.any() else bull_conf_inf

# Reindex to zoom
bear_cand = bear_cand_used.reindex(px_zoom.index).fillna(False)
bear_conf = bear_conf_used.reindex(px_zoom.index).fillna(False)
bull_cand = bull_cand_used.reindex(px_zoom.index).fillna(False)
bull_conf = bull_conf_used.reindex(px_zoom.index).fillna(False)

bear_cand_only = bear_cand & (~bear_conf)
bull_cand_only = bull_cand & (~bull_conf)

# Plot with 4-color shading
fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["Close"],  name="Close",
                             mode="lines", line=dict(width=2.0, color="#111")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.6, color="#ff7f0e")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.6, color="#2ca02c")))
# light/dark RED = bear candidate/confirm
_add_vbands(fig_reg, px_zoom.index, bear_cand_only, "crimson", 0.12)
_add_vbands(fig_reg, px_zoom.index, bear_conf,      "crimson", 0.30)
# light/dark GREEN = bull candidate/confirm
_add_vbands(fig_reg, px_zoom.index, bull_cand_only, "#2ca02c", 0.12)
_add_vbands(fig_reg, px_zoom.index, bull_conf,      "#2ca02c", 0.30)

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
    title=dict(text=f"TSLA — Regimes (last {zoom_years} years; "
                    f"green=bull, red=bear; light=candidate, dark=confirmed)<br>"
                    f"<sup>{params_str}</sup>", pad=dict(b=26)),
    template="plotly_white", height=540,
    margin=dict(l=10, r=10, t=96, b=10),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    hovermode="x unified",
)
st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")

# ── tables: confirmed bear & bull with prices and returns ───────────────────
st.markdown("### Confirmed segments in zoom window")

def _segments_list(index, mask_bool):
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

def _safe_pick(series: pd.Series, ts, default=float("nan")):
    try: return float(series.loc[ts])
    except Exception: return default

def _rows_for(mask: pd.Series, label: str):
    rows = []
    for s, e in _segments_list(px_zoom.index, mask):
        s_close = _safe_pick(px_zoom["Close"], s)
        e_close = _safe_pick(px_zoom["Close"], e)
        ret = (e_close / s_close - 1.0) if (s_close and e_close) else float("nan")
        rows.append({
            "type": label, "start": s, "end": e, "days": (e - s).days + 1,
            "start_close": s_close, "end_close": e_close, "return_%": ret * 100.0
        })
    return rows

rows = _rows_for(bear_conf, "bear_confirm") + _rows_for(bull_conf, "bull_confirm")
seg_df = pd.DataFrame(rows, columns=["type","start","end","days","start_close","end_close","return_%"]).sort_values("start")

if seg_df.empty:
    st.info("No confirmed segments in the zoom window.")
else:
    st.dataframe(seg_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download segments (CSV)",
        data=seg_df.to_csv(index=False).encode("utf-8"),
        file_name=f"tsla_confirmed_segments_last{zoom_years}y.csv",
        mime="text/csv", use_container_width=True
    )
    st.text_area("Quick copy (CSV)", seg_df.to_csv(index=False), height=160)

# ── debug panel (now includes bulls) ────────────────────────────────────────
show_debug = st.sidebar.checkbox("Debug mode (regimes)", value=False)
if show_debug:
    st.markdown("### Regime debug")
    with st.expander("Mask counts on the zoom window", expanded=True):
        st.write({
            "bear_candidate_used.sum": int(bear_cand.reindex(px_zoom.index).sum()),
            "bear_confirmed_used.sum": int(bear_conf.reindex(px_zoom.index).sum()),
            "bull_candidate_used.sum": int(bull_cand.reindex(px_zoom.index).sum()),
            "bull_confirmed_used.sum": int(bull_conf.reindex(px_zoom.index).sum()),
        })
    p_bear = _try_get_prob_series(df) or pd.Series([], dtype=float)
    if not p_bear.empty:
        pz = p_bear.reindex(px_zoom.index)
        pz_ema = pz.ewm(span=ema_span, adjust=False).mean()
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=pz.index, y=pz.values, name="p_bear", mode="lines"))
        fig_prob.add_trace(go.Scatter(x=pz_ema.index, y=pz_ema.values, name=f"EMA({ema_span})", mode="lines"))
        fig_prob.add_hline(y=bear_enter, line=dict(width=1, dash="dash"),
                           annotation_text="enter", annotation_position="top left")
        fig_prob.add_hline(y=bear_exit, line=dict(width=1, dash="dot"),
                           annotation_text="exit", annotation_position="bottom left")
        fig_prob.update_layout(height=260, template="plotly_white", title="Bear probability (raw & EMA)")
        st.plotly_chart(fig_prob, use_container_width=True)

st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
