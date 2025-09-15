# streamlit_app/Home.py
from __future__ import annotations

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PURPOSE                                                                ║
# ║  -------                                                                ║
# ║  Streamlit page with three TSLA charts:                                 ║
# ║    1) IPO→today Close + EMA20/EMA100                                    ║
# ║    2) Last N years Close + EMA20/EMA100                                 ║
# ║    3) Last N years "Regime" view with bull/bear candidate & confirmed   ║
# ║       shading (green=bulllight/dark, red=bear light/dark).              ║
# ║                                                                          ║
# ║  Robust to minor API changes in `detect_regimes` via a flexible adapter.║
# ║  If the pipeline doesn’t provide masks, we infer them sensibly.         ║
# ║                                                                          ║
# ║  Locked (hidden) knobs per your request:                                ║
# ║    - n_components        = 4                                            ║
# ║    - zoom_years          = 3                                            ║
# ║    - entry_ret_lookback  = 10                                           ║
# ║    - entry_ret_thresh    = -0.01                                        ║
# ║    - confirm_days_bull   = 3                                            ║
# ║    - min_bull_run        = 5                                            ║
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
# Import path setup
# ────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ────────────────────────────────────────────────────────────────────────────
# Pipeline import (friendly failure)
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

# ════════════════════════════════════════════════════════════════════════════
#                   LOCKED (HIDDEN) DEMO DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
# We **intentionally hide** these parameters to keep the demo focused.
N_COMPONENTS_LOCK       = 4      # HMM states (locked)
ZOOM_YEARS_LOCK         = 3      # years (locked)
ENTRY_RETLBK_LOCK       = 10     # days (locked)
ENTRY_RETTHR_LOCK       = -0.01  # -1% (locked)
CONFIRM_DAYS_BULL_LOCK  = 3      # (locked)
MIN_BULL_RUN_LOCK       = 5      # days (locked)

# ===========================================================================#
#                                SIDEBAR                                     #
# ===========================================================================#
st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")
st.sidebar.markdown("### Controls (fixed to TSLA)")
ticker = "TSLA"

# Show locked info without offering controls
st.sidebar.info(
    f"**HMM states (locked):** {N_COMPONENTS_LOCK}\n\n"
    f"**Zoom window (locked):** {ZOOM_YEARS_LOCK} years\n\n"
    f"**Other locks:** entry_ret_lookback=10, entry_ret_thresh=-0.01, "
    f"confirm_days_bull=3, min_bull_run=5"
)

# Visible knobs (keep compact & intuitive)
st.sidebar.markdown("### Regime knobs (demo-friendly)")
k_forward           = st.sidebar.slider("k_forward (days ahead for labeling)", 1, 20, value=10, step=1)
ema_span            = st.sidebar.slider("EMA smoothing of bear prob (ema_span)", 5, 60, value=20, step=1)
bear_enter          = st.sidebar.slider("Bear enter threshold", 0.50, 0.99, value=0.80, step=0.01)
bear_exit           = st.sidebar.slider("Bear exit threshold", 0.00, 0.95, value=0.55, step=0.01)

min_bear_run        = st.sidebar.slider("Min bear run (days)", 1, 60, value=15, step=1)
# NOTE: min_bull_run is **hidden** — locked at MIN_BULL_RUN_LOCK

mom_threshold       = st.sidebar.slider("Trend weakness (mom_threshold)", 0.00, 0.10, value=0.03, step=0.001)
ddown_threshold     = st.sidebar.slider("Drawdown confirm (ddown_threshold)", 0.00, 0.30, value=0.15, step=0.005)
confirm_days        = st.sidebar.slider("Confirm days (bear)", 0, 20, value=7, step=1)

# Bull-side tuning (confirm_days_bull is locked)
bull_mom_threshold  = st.sidebar.slider("Bull trend (bull_mom_threshold)", 0.00, 0.05, value=0.01, step=0.001)
bull_ddown_exit     = st.sidebar.slider("Bull dd exit (bull_ddown_exit)", 0.00, 0.20, value=0.06, step=0.005)

direction_gate      = st.sidebar.checkbox("Directional gate", value=True)
trend_gate          = st.sidebar.checkbox("Trend gate", value=True)
strict              = st.sidebar.checkbox("Strict confirmation", value=False)

# Entry drawdown & profit exit: leave visible (optional)
entry_ddown_thresh  = st.sidebar.slider("Entry drawdown threshold", -0.10, 0.10, value=-0.03, step=0.001)
bear_profit_exit    = st.sidebar.slider("Bear profit exit", 0.00, 0.20, value=0.05, step=0.005)

# Apply locks
n_components = N_COMPONENTS_LOCK
zoom_years   = ZOOM_YEARS_LOCK

# ===========================================================================#
#                              HELPERS                                       #
# ===========================================================================#

def _call_detect_regimes_flexible(func, **vals):
    """
    Map the page’s knob names to whatever the current `detect_regimes`
    signature expects. Returns (df, model) even if the original returns
    only a DataFrame.
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
    put(["strict"],                 vals.get("strict"))

    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)


def _ensure_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure ema20/ema100 exist for plotting/confirmation rules."""
    if df is None or df.empty:
        return df
    if "Close" in df:
        if "ema20" not in df.columns:
            df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        if "ema100" not in df.columns:
            df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
    return df


def _last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Tail slice to the last `years` using the DateTimeIndex."""
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[df.index >= start].copy()


def _try_get_prob_series(df: pd.DataFrame, kind: str = "bear") -> pd.Series | None:
    """Find a probability series by fuzzy name: 'p_bear' or 'p_bull' if present."""
    kind = kind.lower()
    # prefer explicit names
    cand = [c for c in df.columns if "prob" in c.lower() and kind in c.lower()]
    if cand:
        s = pd.to_numeric(df[cand[0]], errors="coerce")
        return s.astype(float)
    # convenience aliases
    if kind == "bear" and "p_bear" in df:
        return pd.to_numeric(df["p_bear"], errors="coerce").astype(float)
    if kind == "bull" and "p_bull" in df:
        return pd.to_numeric(df["p_bull"], errors="coerce").astype(float)
    return None


def _min_run_filter(mask: pd.Series, min_len: int) -> pd.Series:
    """Basic run-length cleaner: drop True runs shorter than `min_len`."""
    mask = pd.Series(mask, index=mask.index, dtype=bool)
    run_id = (mask != mask.shift()).cumsum()
    kept = mask.copy()
    for _, seg in mask.groupby(run_id):
        if seg.iloc[0] and len(seg) < min_len:
            kept.loc[seg.index] = False
    return kept


def _infer_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Build bull/bear candidate & confirm masks if missing:
      - Bear candidate: EMA(p_bear) >= bear_enter
      - Bear confirm:  bear_cand & (ema20 < ema100*(1-mom_thr)) & (ddown <= -ddown_thr)
      - Bull candidate: EMA(p_bull) if present else (ema20 > ema100*(1+bull_mom_thr)) | (ddown >= -bull_ddown_exit)
      - Bull confirm:   bull_cand & (ema20 > ema100*(1+bull_mom_thr))  (persistence locked at CONFIRM_DAYS_BULL_LOCK)
    """
    idx = df.index

    # --- Bear side ---
    p_bear = _try_get_prob_series(df, "bear")
    if p_bear is not None:
        bear_cand = (p_bear.ewm(span=ema_span, adjust=False).mean() >= bear_enter).reindex(idx).fillna(False)
    else:
        bear_cand = pd.Series(False, index=idx)

    tmp = _ensure_emas(df.copy())
    ema_bear_ok = (tmp["ema20"] < tmp["ema100"] * (1 - mom_threshold)).reindex(idx).fillna(False)
    if "drawdown" in tmp:
        dd = pd.to_numeric(tmp["drawdown"], errors="coerce")
    else:
        dd = (tmp["Close"] / tmp["Close"].cummax() - 1.0)
    dd_bear_ok = (dd <= -ddown_threshold).fillna(False)

    bear_conf = (bear_cand & ema_bear_ok & dd_bear_ok)
    bear_cand = _min_run_filter(bear_cand, min_bear_run)
    bear_conf = _min_run_filter(bear_conf, max(1, min_bear_run // 2))

    # --- Bull side ---
    p_bull = _try_get_prob_series(df, "bull")
    ema_bull_ok = (tmp["ema20"] > tmp["ema100"] * (1 + bull_mom_threshold)).reindex(idx).fillna(False)
    dd_bull_ok  = (dd >= -bull_ddown_exit).fillna(False)  # drawdown healed

    if p_bull is not None:
        bull_cand = (p_bull.ewm(span=ema_span, adjust=False).mean() >= (1.0 - bear_enter)).reindex(idx).fillna(False)
    else:
        # fallback: momentum or healed drawdown counts as a bull candidate
        bull_cand = (ema_bull_ok | dd_bull_ok)

    bull_conf = _min_run_filter(bull_cand & ema_bull_ok, max(1, CONFIRM_DAYS_BULL_LOCK))
    bull_cand = _min_run_filter(bull_cand, MIN_BULL_RUN_LOCK)

    return {
        "bear_candidate": bear_cand.astype(bool),
        "bear_confirm":   bear_conf.astype(bool),
        "bull_candidate": bull_cand.astype(bool),
        "bull_confirm":   bull_conf.astype(bool),
    }


def _segments(index, mask_bool):
    """Turn a boolean mask into [(start, end), ...] runs."""
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
    Add vertical shaded bands for True runs in `mask_bool`.
    Use add_shape + yref='paper' so the bands span the full plot height
    and never get clipped by y-axis range (robust across Plotly versions).
    """
    for s, e in _segments(idx, mask_bool):
        fig.add_shape(
            type="rect",
            x0=s, x1=e,
            xref="x",
            y0=0, y1=1,
            yref="paper",
            fillcolor=color,
            opacity=opacity,
            line=dict(width=0),
            layer="below",
        )


def _segment_table(mask: pd.Series, close: pd.Series, label: str) -> pd.DataFrame:
    """Build a copyable table of (type, start, end, days, start_close, end_close, return)."""
    rows = []
    for s, e in _segments(mask.index, mask.values):
        start_close = float(close.loc[s])
        end_close   = float(close.loc[e])
        ret = (end_close / start_close - 1.0) if start_close else float("nan")
        rows.append({
            "type": label,
            "start": s.strftime("%Y-%m-%d"),
            "end":   e.strftime("%Y-%m-%d"),
            "days":  (e - s).days + 1,
            "start_close": round(start_close, 4),
            "end_close":   round(end_close, 4),
            "return":      round(ret, 6),
        })
    return pd.DataFrame(rows, columns=["type","start","end","days","start_close","end_close","return"])


# ===========================================================================#
#                     RUN PIPELINE (ONCE) AND PLOT                           #
# ===========================================================================#
with st.spinner("Running regime pipeline (TSLA)…"):
    # Pass the **locked** values where appropriate.
    df, _ = _call_detect_regimes_flexible(
        detect_regimes,
        ticker="TSLA", start="2000-01-01", end="today",
        n_components=n_components,                 # ← locked (4)
        k_forward=k_forward,
        ema_span=ema_span,
        bear_enter=bear_enter,
        bear_exit=bear_exit,
        min_bear_run=min_bear_run,
        min_bull_run=MIN_BULL_RUN_LOCK,           # ← locked (5)
        mom_threshold=mom_threshold,
        ddown_threshold=ddown_threshold,
        confirm_days=confirm_days,
        bull_mom_threshold=bull_mom_threshold,
        bull_ddown_exit=bull_ddown_exit,
        confirm_days_bull=CONFIRM_DAYS_BULL_LOCK, # ← locked (3)
        direction_gate=direction_gate,
        trend_gate=trend_gate,
        entry_ret_lookback=ENTRY_RETLBK_LOCK,     # ← locked (10)
        entry_ret_thresh=ENTRY_RETTHR_LOCK,       # ← locked (-0.01)
        entry_ddown_thresh=entry_ddown_thresh,
        bear_profit_exit=bear_profit_exit,
        strict=strict,
    )

if df is None or df.empty or "Close" not in df.columns:
    st.error("Pipeline returned no data or missing 'Close'.")
    st.stop()

df = df.sort_index()
df = _ensure_emas(df)

# Build zoom slice from pipeline output (keeps everything aligned)
px_full = df[["Close","ema20","ema100"]].copy()
px_zoom = _last_years(px_full, zoom_years)

# ────────────────────────────────────────────────────────────────────────────
# Charts 1 & 2 (always visible)
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

fig1 = _plot_close_emas(px_full, "TSLA — Close with EMA20 / EMA100 (IPO → today)")
st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

fig2 = _plot_close_emas(px_zoom, f"TSLA — Close with EMA20 / EMA100 (last {zoom_years} years)")
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")

# ────────────────────────────────────────────────────────────────────────────
# 3) Regime view (zoom window) with bull & bear shading
# ────────────────────────────────────────────────────────────────────────────

# Prefer pipeline masks if they are present & have any True; otherwise infer.
bear_cand_raw = pd.Series(df.get("bear_candidate", False), index=df.index).astype(bool)
bear_conf_raw = pd.Series(df.get("bear_confirm",   False), index=df.index).astype(bool)
bull_cand_raw = pd.Series(df.get("bull_candidate", False), index=df.index).astype(bool)
bull_conf_raw = pd.Series(df.get("bull_confirm",   False), index=df.index).astype(bool)

# Build inferred masks
masks_inf = _infer_masks(df)

def _choose(pipeline: pd.Series, inferred: pd.Series) -> pd.Series:
    return pipeline if pipeline.any() else inferred

bear_cand_used = _choose(bear_cand_raw, masks_inf["bear_candidate"])
bear_conf_used = _choose(bear_conf_raw, masks_inf["bear_confirm"])
bull_cand_used = _choose(bull_cand_raw, masks_inf["bull_candidate"])
bull_conf_used = _choose(bull_conf_raw, masks_inf["bull_confirm"])

# Reindex to zoom & build candidate-only versions (so confirmed shading doesn’t double paint)
bear_cand = bear_cand_used.reindex(px_zoom.index).fillna(False)
bear_conf = bear_conf_used.reindex(px_zoom.index).fillna(False)
bull_cand = bull_cand_used.reindex(px_zoom.index).fillna(False)
bull_conf = bull_conf_used.reindex(px_zoom.index).fillna(False)

bear_cand_only = (bear_cand & (~bear_conf))
bull_cand_only = (bull_cand & (~bull_conf))

# --- Regime plot ---
fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["Close"],  name="Close",
                             mode="lines", line=dict(width=2.0, color="#111")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema20"], name="EMA20",
                             mode="lines", line=dict(width=1.6, color="#ff7f0e")))
fig_reg.add_trace(go.Scatter(x=px_zoom.index, y=px_zoom["ema100"], name="EMA100",
                             mode="lines", line=dict(width=1.6, color="#2ca02c")))

# Shading: bull (green), bear (red); light=candidate-only, dark=confirmed
_add_vbands(fig_reg, px_zoom.index, bull_cand_only, "#2ca02c", 0.12)  # light green
_add_vbands(fig_reg, px_zoom.index, bull_conf,      "#2ca02c", 0.30)  # dark  green
_add_vbands(fig_reg, px_zoom.index, bear_cand_only, "#d62728", 0.12)  # light red
_add_vbands(fig_reg, px_zoom.index, bear_conf,      "#d62728", 0.30)  # dark  red

params_str = (
    f"k_fwd={k_forward}, EMA={ema_span}, enter={bear_enter:.2f}, exit={bear_exit:.2f}, "
    f"min_bear={min_bear_run}, min_bull={MIN_BULL_RUN_LOCK}, mom_thr={mom_threshold:.2f}, "
    f"dd_thr={ddown_threshold:.2f}, bull_mom_thr={bull_mom_threshold:.2f}, "
    f"bull_dd_exit={bull_ddown_exit:.2f}, confirm_bear={confirm_days}, "
    f"confirm_bull={CONFIRM_DAYS_BULL_LOCK}, dir_gate={direction_gate}, "
    f"lbk={ENTRY_RETLBK_LOCK}, entry_ret_thr={ENTRY_RETTHR_LOCK:.2f}, "
    f"entry_dd_thr={entry_ddown_thresh:.2f}, trend_gate={trend_gate}, "
    f"profit_exit={bear_profit_exit:.2f}, strict={strict}"
)
fig_reg.update_layout(
    title=dict(text=f"TSLA — Regimes (last {zoom_years} years; green=bull, red=bear; light=candidate, dark=confirmed)<br>"
                    f"<sup>{params_str}</sup>", pad=dict(b=26)),
    template="plotly_white",
    height=540,
    margin=dict(l=10, r=10, t=96, b=10),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    hovermode="x unified",
)
st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")

# ────────────────────────────────────────────────────────────────────────────
# Confirmed segments tables (copyable)
# ────────────────────────────────────────────────────────────────────────────
st.markdown("### Confirmed segments in zoom window")

bull_table = _segment_table(bull_conf, px_zoom["Close"], "bull_confirm")
bear_table = _segment_table(bear_conf, px_zoom["Close"], "bear_confirm")

colA, colB = st.columns(2)
with colA:
    st.subheader("Bull — confirmed")
    if bull_table.empty:
        st.info("No confirmed bull segments in the zoom window.")
    else:
        st.dataframe(bull_table, use_container_width=True)
with colB:
    st.subheader("Bear — confirmed")
    if bear_table.empty:
        st.info("No confirmed bear segments in the zoom window.")
    else:
        st.dataframe(bear_table, use_container_width=True)

# ===========================================================================#
#                               DEBUG PANEL                                  #
# ===========================================================================#
show_debug = st.sidebar.checkbox("Debug mode (regimes)", value=False)

if show_debug:
    st.markdown("### Regime debug")
    candidate_source = {
        "bear": "pipeline" if bear_cand_raw.any() else "inferred",
        "bull": "pipeline" if bull_cand_raw.any() else "inferred",
    }
    confirm_source = {
        "bear": "pipeline" if bear_conf_raw.any() else "inferred",
        "bull": "pipeline" if bull_conf_raw.any() else "inferred",
    }

    with st.expander("Input/columns snapshot", expanded=True):
        st.write({
            "df.shape": df.shape,
            "index.range": f"{df.index.min()} → {df.index.max()}",
            "has_bear_candidate_col": bool("bear_candidate" in df.columns),
            "has_bear_confirm_col":  bool("bear_confirm" in df.columns),
            "has_bull_candidate_col": bool("bull_candidate" in df.columns),
            "has_bull_confirm_col":  bool("bull_confirm" in df.columns),
            "pipeline_has_bear_cand_true": bool(bear_cand_raw.any()),
            "pipeline_has_bear_conf_true": bool(bear_conf_raw.any()),
            "pipeline_has_bull_cand_true": bool(bull_cand_raw.any()),
            "pipeline_has_bull_conf_true": bool(bull_conf_raw.any()),
            "candidate_source": candidate_source,
            "confirm_source":   confirm_source,
        })
        st.write("Columns:", list(df.columns)[:60])

    with st.expander("Mask counts on the zoom window", expanded=True):
        st.write({
            "bull_candidate.sum": int(bull_cand.sum()),
            "bull_confirm.sum":   int(bull_conf.sum()),
            "bear_candidate.sum": int(bear_cand.sum()),
            "bear_confirm.sum":   int(bear_conf.sum()),
        })

st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
