# streamlit_app/pages/2_Price_Prediction_with_Regimes.py
from __future__ import annotations

# ---- stdlib / path setup ----
import sys, os, inspect, importlib
from pathlib import Path

# ---- third-party ----
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# ========= repo import path (no edits to existing code) =========
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ========= unchanged pipeline =========
from src.regime_detection import detect_regimes

# ========= regime bias (reload to pick up latest shape-safe version) =========
import addons.regime_bias as _rb
importlib.reload(_rb)  # ensure we use the updated function definition
from addons.regime_bias import apply_regime_bias


# ========= Streamlit page =========
st.set_page_config(page_title="Regime-aware Forecast (TSLA demo)", layout="wide")
st.title("Regime-aware Price Forecast — TSLA (add-on demo)")
st.caption("This page is additive. It reuses the existing pipeline and does not modify the original app.")

# ---------------- Sidebar: locked to TSLA + familiar knobs ----------------
st.sidebar.header("Controls (TSLA demo)")
ticker = "TSLA"
zoom_years = st.sidebar.slider("Zoom window (years)", 1, 10, 3, 1)

st.sidebar.subheader("Regime knobs (same defaults)")
n_components        = st.sidebar.number_input("HMM states", 2, 6, 4, 1)
k_forward           = st.sidebar.slider("k_forward", 1, 20, 10, 1)
ema_span            = st.sidebar.slider("ema_span", 5, 60, 20, 1)
bear_enter          = st.sidebar.slider("bear_enter", 0.50, 0.99, 0.80, 0.01)
bear_exit           = st.sidebar.slider("bear_exit", 0.00, 0.95, 0.55, 0.01)
min_bear_run        = st.sidebar.slider("min_bear_run (days)", 1, 60, 15, 1)
min_bull_run        = st.sidebar.slider("min_bull_run (days)", 1, 60, 5, 1)
mom_threshold       = st.sidebar.slider("mom_threshold", 0.00, 0.10, 0.03, 0.001)
ddown_threshold     = st.sidebar.slider("ddown_threshold", 0.00, 0.30, 0.15, 0.005)
confirm_days        = st.sidebar.slider("confirm_days (bear)", 0, 20, 7, 1)
bull_mom_threshold  = st.sidebar.slider("bull_mom_threshold", 0.00, 0.05, 0.01, 0.001)
bull_ddown_exit     = st.sidebar.slider("bull_ddown_exit", 0.00, 0.20, 0.06, 0.005)
confirm_days_bull   = st.sidebar.slider("confirm_days (bull)", 0, 10, 3, 1)
direction_gate      = st.sidebar.checkbox("direction_gate", True)
trend_gate          = st.sidebar.checkbox("trend_gate", True)
strict              = st.sidebar.checkbox("strict", False)
entry_ret_lookback  = st.sidebar.slider("entry_ret_lookback", 1, 30, 10, 1)
entry_ret_thresh    = st.sidebar.slider("entry_ret_thresh", -0.05, 0.05, -0.01, 0.001)
entry_ddown_thresh  = st.sidebar.slider("entry_ddown_thresh", -0.10, 0.10, -0.03, 0.001)
bear_profit_exit    = st.sidebar.slider("bear_profit_exit", 0.00, 0.20, 0.05, 0.005)

st.sidebar.subheader("Bias layer")
vol_span     = st.sidebar.slider("vol_span (for bias σ)", 5, 60, 20, 1)
bull_k       = st.sidebar.slider("bull_k (×σ)", -1.0, 2.0, 0.60, 0.05)
bear_k_conf  = st.sidebar.slider("bear_k_conf (×σ)", -2.0, 0.0, -0.60, 0.05)
bear_k_cand  = st.sidebar.slider("bear_k_cand (×σ)", -2.0, 0.0, -0.30, 0.05)

with st.sidebar.expander("ARIMA baseline (advanced)"):
    p = st.number_input("ARIMA p", 0, 5, value=1, step=1)
    d = st.number_input("ARIMA d", 0, 2, value=1, step=1)
    q = st.number_input("ARIMA q", 0, 5, value=1, step=1)

# ========================= Helpers =========================
def _clean_series(s: pd.Series) -> pd.Series:
    """Coerce to float, drop NaNs, ensure 1-D series."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    return s.astype(float).dropna()

def _align_like(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reindex b to a's index; drop rows where either is NaN."""
    b2 = b.reindex(a.index)
    mask = ~(a.isna() | b2.isna())
    return a[mask], b2[mask]

def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """RMSE without extra deps."""
    y_true, y_pred = y_true.align(y_pred, join="inner")
    y_true = pd.to_numeric(y_true, errors="coerce").astype(float)
    y_pred = pd.to_numeric(y_pred, errors="coerce").astype(float)
    m = ~(y_true.isna() | y_pred.isna())
    n = int(m.sum())
    if n < 5:
        return np.nan
    diff = (y_true[m] - y_pred[m]).values
    return float(np.sqrt(np.mean(diff * diff)))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv_last_years(ticker: str = "TSLA", years: int = 3) -> pd.DataFrame:
    """OHLCV for last N years (auto-adjusted)."""
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close_full(ticker: str = "TSLA") -> pd.DataFrame:
    """Full max history Close for slicing the zoom window index."""
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=True, progress=False)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Close"]].dropna()

def base_forecast_close_arima(ohlcv: pd.DataFrame, order=(1, 1, 1)) -> pd.Series:
    """
    One-step-ahead ARIMA baseline for the *historical* Close series.
    Returns a Series aligned to ohlcv.index named 'base_forecast'.
    """
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv:
        return pd.Series(dtype=float, name="base_forecast")

    close = ohlcv["Close"].astype(float)
    if len(close) < 50:
        return close.rename("base_forecast")

    try:
        model = ARIMA(close, order=order)
        res = model.fit()
        # one-step-ahead predicted mean; align to same index (fill first)
        pred = res.get_prediction(start=close.index[1], end=close.index[-1], dynamic=False)
        base = pred.predicted_mean.reindex(close.index)
        base.iloc[0] = close.iloc[0]
        return base.rename("base_forecast")
    except Exception as e:
        st.warning(f"ARIMA baseline failed ({e}). Falling back to a simple EMA(10).")
        return close.ewm(span=10, adjust=False).mean().rename("base_forecast")

def _call_detect_regimes(func, **vals):
    """Flexible mapper to whatever signature your detect_regimes exposes."""
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    out = {}
    def put(names, val):
        for n in names:
            if n in params:
                out[n] = val; return
    put(["ticker"],               vals["ticker"])
    put(["start"],                vals.get("start"))
    put(["end"],                  vals.get("end"))
    put(["n_components"],         vals["n_components"])
    put(["k_forward","k_fwd"],    vals["k_forward"])
    put(["ema_span","ema"],       vals["ema_span"])
    put(["bear_enter","enter_threshold","prob_threshold"], vals["bear_enter"])
    put(["bear_exit","exit_threshold","prob_exit"],        vals["bear_exit"])
    put(["min_bear_run","min_run"], vals["min_bear_run"])
    put(["min_bull_run"],          vals["min_bull_run"])
    put(["mom_threshold","mom_thr"], vals["mom_threshold"])
    put(["ddown_threshold","dd_thr"], vals["ddown_threshold"])
    put(["confirm_days","confirm_bear"], vals["confirm_days"])
    put(["bull_mom_threshold","bull_mom_thr"], vals["bull_mom_threshold"])
    put(["bull_ddown_exit","bull_dd_exit"], vals["bull_ddown_exit"])
    put(["confirm_days_bull","confirm_bull"], vals["confirm_days_bull"])
    put(["direction_gate"], vals["direction_gate"])
    put(["trend_gate"],     vals["trend_gate"])
    put(["entry_ret_lookback","lbk","lookback"], vals["entry_ret_lookback"])
    put(["entry_ret_thresh","entry_ret_thr"],    vals["entry_ret_thresh"])
    put(["entry_ddown_thresh","entry_dd_thr"],   vals["entry_ddown_thresh"])
    put(["bear_profit_exit","profit_exit"],      vals["bear_profit_exit"])
    put(["strict"],           vals["strict"])
    res = func(**out)
    return res if isinstance(res, tuple) else (res, None)

# ========================= Data & pipeline =========================
with st.spinner("Fetching TSLA & detecting regimes…"):
    # Zoom index driven by full Close history
    px_full = fetch_close_full(ticker)
    cutoff  = px_full.index.max() - pd.DateOffset(years=zoom_years)
    px_zoom = px_full.loc[px_full.index >= cutoff].copy()

    # Unchanged pipeline over full history, reindexed to the zoom window
    df_reg, _ = _call_detect_regimes(
        detect_regimes,
        ticker=ticker, start="2000-01-01", end="today",
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
    reg_zoom = df_reg.reindex(px_zoom.index)

# Normalize dtypes to strict booleans (avoids array/nested types)
for col in ["bear_candidate", "bear_confirm", "bull_candidate", "bull_confirm"]:
    if col in reg_zoom.columns:
        reg_zoom[col] = reg_zoom[col].astype(bool)

# ========================= ARIMA baseline on OHLCV (last N years) =========================
ohlcv = fetch_ohlcv_last_years(ticker, years=zoom_years)
if ohlcv.empty:
    st.error("Could not fetch OHLCV for ARIMA baseline. Try rerunning.")
    st.stop()

base     = base_forecast_close_arima(ohlcv, order=(p, d, q))     # Series
base     = pd.Series(base, index=base.index, dtype="float64")
px_zoom  = px_zoom.reindex(base.index)                           # align views to base
reg_zoom = reg_zoom.reindex(base.index)

# Final alignment cleanup
close = _clean_series(px_zoom["Close"])
close, base = _align_like(close, base)

# ========================= Regime-aware bias =========================
final, bias = apply_regime_bias(
    base=base,
    close=close,
    regime_df=reg_zoom,
    vol_span=vol_span,
    bull_k=bull_k,
    bear_k_conf=bear_k_conf,
    bear_k_cand=bear_k_cand,
)

# Align all three to the same index (and drop any residual NaNs)
close, base = _align_like(close, base)
close, final = _align_like(close, final)

if min(len(close), len(base), len(final)) < 5:
    st.warning("Not enough aligned points to plot after cleaning/alignment.")
    st.stop()

# ========================= Hybrid via confirmed windows (CSV from Home.py) =========================
seg_csv = Path(REPO_ROOT, "reports", "confirmed_segments_last3y.csv")
confirmed_windows: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
if seg_csv.exists():
    try:
        seg_raw = pd.read_csv(seg_csv)
        seg_raw = seg_raw[seg_raw["type"].isin(["bull_confirm", "bear_confirm"])].copy()
        seg_raw["start"] = pd.to_datetime(seg_raw["start"])
        seg_raw["end"]   = pd.to_datetime(seg_raw["end"])
        confirmed_windows = [(r["start"], r["end"], r["type"]) for _, r in seg_raw.iterrows()]
        st.caption(f"Loaded confirmed windows from `{seg_csv}`")
    except Exception as e:
        st.warning(f"Could not read confirmed segments CSV ({e}). Proceeding without hybrid swap.")
else:
    st.info("Confirmed segments CSV not found. Generate it by opening the Home page once (it saves last-3y confirmed segments).")

# Hybrid: ARIMA except replaced by regime-aware during confirmed bull/bear windows
hybrid = base.copy()
if confirmed_windows:
    idx_min, idx_max = close.index.min(), close.index.max()
    for s, e, typ in confirmed_windows:
        s2 = max(pd.Timestamp(s), idx_min)
        e2 = min(pd.Timestamp(e), idx_max)
        if s2 <= e2:
            mask = (hybrid.index >= s2) & (hybrid.index <= e2)
            # Replace with regime-aware forecast for those dates
            hybrid.loc[mask] = final.loc[mask]

# ========================= RMSEs =========================
rmse_arima  = _rmse(close, base)
rmse_hybrid = _rmse(close, hybrid)

# ========================= Plot (Actual vs ARIMA vs Hybrid) =========================
fig = go.Figure()

# Background shading to show confirmed windows (green for bull, red for bear)
ymin = float(close.min()) * 0.95
ymax = float(close.max()) * 1.05
for s, e, typ in confirmed_windows:
    s2 = max(pd.Timestamp(s), close.index.min())
    e2 = min(pd.Timestamp(e), close.index.max())
    if s2 <= e2:
        fig.add_vrect(
            x0=s2, x1=e2,
            fillcolor=("#2ca02c" if typ == "bull_confirm" else "#d62728"),
            opacity=0.16, line_width=0, layer="below"
        )

# Lines
fig.add_trace(go.Scatter(
    x=close.index, y=close.values, name="Actual (Close)",
    mode="lines", line=dict(width=2.0, color="#111")
))
fig.add_trace(go.Scatter(
    x=base.index, y=base.values,
    name=(f"ARIMA({p},{d},{q}) — RMSE={rmse_arima:.3f}" if pd.notna(rmse_arima) else f"ARIMA({p},{d},{q})"),
    mode="lines", line=dict(width=2.0, dash="dash", color="#d62728")
))
fig.add_trace(go.Scatter(
    x=hybrid.index, y=hybrid.values,
    name=(f"Hybrid (Regime-aware inside confirmed windows) — RMSE={rmse_hybrid:.3f}"
          if pd.notna(rmse_hybrid) else "Hybrid (Regime-aware inside confirmed windows)"),
    mode="lines", line=dict(width=2.4, color="#1f77b4")
))

fig.update_layout(
    title=f"TSLA — Actual vs ARIMA vs Hybrid (last {zoom_years}y; shaded = confirmed bull/bear windows)",
    template="plotly_white",
    height=560,
    margin=dict(l=10, r=10, t=90, b=10),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.85)"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", range=[ymin, ymax]),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with st.expander("What’s happening here?", expanded=False):
    st.markdown(
        f"""
**Baseline:** ARIMA one-step-ahead projection on last-{zoom_years} years daily **Close** (built from OHLCV).  
**Regime-aware:** Baseline tilted by regime + recent volatility (σ).  
**Hybrid:** Use ARIMA everywhere, but **swap to regime-aware** inside **confirmed bull/bear windows** saved by the Home page.

- RMSE(ARIMA): **{rmse_arima:.3f}**  
- RMSE(Hybrid): **{rmse_hybrid:.3f}**

Shaded bands show the windows where the Hybrid uses regime-aware values:
green = confirmed bull, red = confirmed bear.
        """
    )

# ===================== Parameter explainer (bottom of page) =====================
with st.container():
    st.markdown("---")
    st.subheader("Parameter explainer")

    st.markdown("""
**How to read this:** each knob says *what it does*, *when to nudge it up/down*, and a **TSLA-flavored example** so it’s easy to picture.

### General (view)
- **Zoom window (years)** – how many years you see in the plots.  
  *Nudge:* Increase to show more history; decrease to zoom in on recent behavior.  
  *TSLA tip:* 3y shows recent cycles without losing context.

### HMM / Regime labeling (shared with the main app)
- **HMM states (n_components)** – how many hidden market “modes” the HMM learns (e.g., calm-bull, high-vol bull, drawdown-bear).  
  *Nudge:* Raise if one state mixes very different behavior; lower if two states look redundant.  
  *TSLA tip:* 3–4 often separates “fast rising but noisy” from “true damage”.

- **k_forward (days)** – a look-ahead used only while training to name states from future returns.  
  *Nudge:* Larger = smoother, slower labels; smaller = faster, noisier labels.  
  *TSLA tip:* 7–10 captures turns without over-reacting to one-day spikes.

- **ema_span (days)** – smoothing applied to the bear-probability line before thresholds.  
  *Nudge:* Raise to reduce whipsaws; lower to react faster.  
  *TSLA tip:* ~20 keeps noise out but still catches regime changes within weeks.

- **Bear enter / exit (bear_enter, bear_exit)** – probabilities that gate when a **bear candidate** starts and when a bear ends.  
  *Nudge:* If you see too many false bears, raise **enter** or lower **exit**; if bears start too late, lower **enter** a touch.  
  *TSLA tip:* 0.80 enter with 0.55 exit gives healthy hysteresis.

- **Trend weakness (mom_threshold)** – requires **EMA20 below EMA100** by at least this fraction to *confirm* a bear.  
  *Nudge:* Raise to demand a clearer down-trend; lower if bears aren’t confirming when they should.  
  *Example:* 0.03 ≈ 3% gap (EMA20 under EMA100).

- **Drawdown confirm (ddown_threshold)** – price must be this far under a recent peak to *confirm* a bear.  
  *Nudge:* Raise to avoid shallow dips being called bear; lower to catch pullbacks earlier.  
  *TSLA tip:* 0.15 (~15%) filters routine wobbles.

- **Confirm days (bear) (confirm_days)** – how long weakness must persist to confirm bear.  
  *Nudge:* Raise to avoid whipsaws; lower to react sooner.  
  *TSLA tip:* ~7 days.

- **Min run lengths (min_bear_run, min_bull_run)** – drop tiny islands shorter than these durations.  
  *Nudge:* Raise to tidy labels; lower to show more micro-pockets.

- **Bull confirmations (bull_mom_threshold, confirm_days_bull)** – ask for **EMA20 above EMA100** by this fraction and for at least N days before recognizing a bull pocket.  
  *TSLA tip:* 0.01 and 2–3 days catch relief rallies without over-labeling.

- **Bull drawdown exit (bull_ddown_exit)** – if the drawdown has healed to within this distance of the peak, exit bear even if probability lags.  
  *Nudge:* Raise to exit bears earlier after strong recoveries.

- **Gates at entry/exit**
  - **direction_gate** – require weak recent returns **and** a real drawdown to *start* a bear (e.g., last 10-day return ≤ −1% and drawdown ≤ −3%).  
  - **trend_gate** – only enter bear if price is already **below EMA100**.  
  - **trend_exit_cross** (if present) – if price crosses **above EMA100** while in bear, exit.

- **Entry filters** (if shown)
  - **entry_ret_lookback / entry_ret_thresh** – recent return window and threshold to allow a bear entry (e.g., 10 days, −1%).  
  - **entry_ddown_thresh** – minimum drawdown needed to allow bear entry.  
  - **bear_profit_exit** – if price bounces X% from bear entry, force an early bear exit.

### Bias layer (new on this page, for the regime-aware forecast)
- **Volatility span (vol_span)** – EMA window used for a simple vol proxy. Higher span = smoother vol estimate.  
  *Why it matters:* We dampen or amplify the forecast using both *regime* and *vol*.  

- **Bias strength in bull (bull_k)** – how much to **tilt the base forecast up** during bull regimes.  
  *Nudge:* Raise if bull periods under-predict; lower if they overshoot.  
  *TSLA tip:* Start small (e.g., 0.15) and increase if your base model is conservative.

- **Bias strength in confirmed bear (bear_k_conf)** – how much to **tilt the base forecast down** during confirmed bear.  
  *Nudge:* Raise if confirmed bears still look too optimistic; lower if too pessimistic.

- **Bias strength in candidate bear (bear_k_cand)** – gentler tilt while the bear is *only a candidate*.  
  *Nudge:* Keep lower than bear_k_conf so candidates don’t dominate; lift slightly if candidates often turn into real bears.

### What to tweak for different personalities
- **Very volatile (TSLA-like):** keep ema_span ~20, bear_enter high (≈0.80), bear_exit well lower (≈0.55), require mom_threshold ≈3% and ddown_threshold ≈15%, confirm_days ~7, enable direction_gate. Use modest bull_k, stronger bear_k_conf, small bear_k_cand.  
- **Steadier mega-caps:** relax confirms a bit (e.g., mom_threshold 1–2%, ddown_threshold 8–10%), fewer confirm days (~3–5), smaller min runs.  
- **Choppy sideways:** lower bear_enter/raise bear_exit gap slightly and increase min runs to avoid label flip-flop; reduce bias strengths so forecast stays close to the base model.

**Rule of thumb:** If you see too many false bears → raise bear_enter, mom_threshold, ddown_threshold, confirm_days; keep direction_gate on. If bears exit too late → lower bear_exit a bit, raise bull_ddown_exit, or turn on trend_exit_cross.
""")
