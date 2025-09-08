# streamlit_app/Home.py — uses pipeline data for ALL charts + crisp Plotly
from datetime import date
import inspect
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="TSLA Regime Detection", layout="wide")
st.title("TSLA Regime Detection — HMM + Human-Readable Rules (Crisp Zoom Charts)")
st.caption("Full-history price (IPO→today) and last-N-years regime view. Data fetched live via the pipeline (no second download).")

# ---- import your pipeline ----
try:
    from src.regime_detection import detect_regimes
    PIPELINE_IMPORT_ERROR = None
except Exception as e:
    detect_regimes = None
    PIPELINE_IMPORT_ERROR = e

if detect_regimes is None:
    st.error(
        "Couldn't import pipeline: `from src.regime_detection import detect_regimes` failed.\n\n"
        f"Error: {PIPELINE_IMPORT_ERROR}\n"
        "Ensure src/regime_detection.py defines detect_regimes()."
    )
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")

with st.sidebar.expander("General", expanded=True):
    n_components = st.number_input("HMM states (n_components)", 2, 6, 4, 1)
    ema_span     = st.slider("Smoothing span for bear probability (ema_span)", 5, 60, 20, 1)
    k_forward    = st.slider("Horizon to judge state behavior (k_forward)", 1, 30, 10, 1)

with st.sidebar.expander("Hysteresis thresholds", expanded=True):
    bear_enter = st.slider("Enter bear when smoothed prob ≥ (bear_enter)", 0.50, 0.99, 0.80, 0.01)
    bear_exit  = st.slider("Exit bear when smoothed prob ≤ (bear_exit)", 0.10, 0.90, 0.55, 0.01)

with st.sidebar.expander("Bear confirmations (downside checks)", expanded=True):
    mom_threshold   = st.slider("Momentum threshold (EMA20 under EMA100 by ~X of price)", 0.0, 0.10, 0.03, 0.001)
    ddown_threshold = st.slider("Drawdown threshold from recent peak", 0.0, 0.50, 0.15, 0.01)
    confirm_days    = st.slider("Confirm days (bear)", 1, 20, 7, 1)

with st.sidebar.expander("Bull confirms & early exits", expanded=False):
    bull_mom_threshold = st.slider("Bull momentum threshold (EMA20 over EMA100 by ~X)", 0.0, 0.10, 0.01, 0.001)
    bull_ddown_exit    = st.slider("Bull drawdown exit (distance to peak shrinks to)", 0.0, 0.20, 0.06, 0.01)
    confirm_days_bull  = st.slider("Confirm days (bull)", 1, 10, 3, 1)
    bear_profit_exit   = st.slider("Bounce/Profit exit from bear (+X from entry)", 0.0, 0.20, 0.05, 0.005)

with st.sidebar.expander("Gates & cleanup", expanded=False):
    direction_gate     = st.checkbox("Require recent weakness at entry (direction_gate)", True)
    entry_ret_lookback = st.slider("Lookback L for entry direction (days)", 5, 30, 10, 1)
    entry_ret_thresh   = st.slider("Cumulative return over L days must be ≤", -0.10, 0.05, -0.01, 0.001)
    entry_dd_thresh    = st.slider("Current drawdown must be ≤", -0.30, 0.00, -0.03, 0.005)
    trend_gate         = st.checkbox("Require under-trend at bear entry (trend_gate)", False)
    trend_exit_cross   = st.checkbox("Exit bear on trend cross up", True)
    min_bear_run       = st.slider("Min confirmed-bear run length (days)", 1, 40, 15, 1)
    min_bull_run       = st.slider("Min bull run length (days)", 1, 20, 5, 1)

with st.sidebar.expander("Auto thresholds (optional, adaptive)", expanded=False):
    auto_thresholds   = st.checkbox("Auto-pick bear_enter/bear_exit from recent history", False)
    bear_target       = st.slider("Target bear share (recent window)", 0.05, 0.60, 0.32, 0.01)
    auto_window_years = st.slider("Auto window (years)", 2, 10, 5, 1)
    min_gap           = st.slider("Min gap between enter and exit", 0.02, 0.40, 0.10, 0.01)

with st.sidebar.expander("Figure settings", expanded=False):
    zoom_years     = st.slider("Zoom window (last N years)", 1, 5, 3, 1)

st.sidebar.info("ℹ️ Light red = candidate bear (probability crossed ENTER). Dark red = confirmed bear (weak trend/drawdown persisted).")

# ---------------- Safe pipeline call (always fetch full history here) ----------------
ticker = "TSLA"

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
        end="today",          # explicit label to avoid 'None' logs
        full_history=True     # if your function supports it
    )

    # Back-compat
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

    filtered = {k: v for k, v in pref.items() if k in params}
    return detect_regimes(**filtered)

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

# Full history and zoom slice are now both from the *same* df
df_full = df[["Close", "sma20", "sma100"]].dropna().copy()
cutoff = df_full.index.max() - pd.DateOffset(years=int(zoom_years))
dfz = df_full[df_full.index >= cutoff].copy()

span_full = (df_full.index.max() - df_full.index.min()).days/365.25
st.caption(f"Diagnostic — full-history span: {span_full:.1f}y; zoom window: {int(zoom_years)}y")

# ---------------- Plot helpers (Plotly) ----------------
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToAdd": ["toImage", "drawline", "eraseshape"]}

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
    fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"],  name=f"{name_prefix} Close",
                             mode="lines", line=dict(width=1.8, color="black")))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["sma20"],  name="EMA20 (fast)",
                             mode="lines", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["sma100"], name="EMA100 (slow)",
                             mode="lines", line=dict(width=1.2)))

def add_vrect_segments(fig, dates, mask, rgba, legend_name):
    if mask is None or not bool(mask.any()):
        return
    # legend proxy
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=rgba.replace("0.22","1.0").replace("0.45","1.0"), width=10),
                             name=legend_name, showlegend=True))
    in_run, start = False, None
    vals = mask.values
    for i, on in enumerate(vals):
        if on and not in_run:
            in_run, start = True, dates[i]
        last = (i == len(vals) - 1)
        if in_run and (not on or last):
            end = dates[i]
            fig.add_vrect(x0=start, x1=end, fillcolor=rgba, line_width=0, layer="below")
            in_run, start = False, None

# ---------------- Plot A: Full history Close + EMAs (IPO→today) ----------------
fig_full = figure_base("TSLA — Close with EMA20 / EMA100 (IPO → today)")
add_close_ema(fig_full, df_full)
st.plotly_chart(fig_full, use_container_width=True, config=PLOT_CONFIG)

# ---------------- Plot B: Close + EMAs (last N years) ----------------
fig_last = figure_base(f"TSLA — Close with EMA20 / EMA100 (last {int(zoom_years)} years)")
add_close_ema(fig_last, dfz)
st.plotly_chart(fig_last, use_container_width=True, config=PLOT_CONFIG)

# ---------------- Plot C: Regimes (last N years) ----------------
fig_reg = figure_base(f"TSLA — Regimes (last {int(zoom_years)} years)")
add_close_ema(fig_reg, dfz)

# Build candidate/confirmed masks from pipeline columns
conf_z = (dfz["regime"] == 1) if "regime" in dfz.columns else None
cand_z = None
if "p_bear_ema" in dfz.columns:
    cand_z = dfz["p_bear_ema"] >= float(bear_enter)
    if "bear_confirm" in dfz.columns:
        cand_z = cand_z & (~dfz["bear_confirm"].astype(bool))
    if conf_z is not None:
        cand_z = cand_z & (~conf_z)
elif "bear_candidate" in dfz.columns:
    cand_z = dfz["bear_candidate"].astype(bool)

if cand_z is not None and cand_z.any():
    add_vrect_segments(fig_reg, dfz.index, cand_z, "rgba(255,0,0,0.22)", "Bear (candidate)")
if conf_z is not None and conf_z.any():
    add_vrect_segments(fig_reg, dfz.index, conf_z, "rgba(255,0,0,0.45)", "Bear (confirmed)")

st.plotly_chart(fig_reg, use_container_width=True, config=PLOT_CONFIG)

# ---------------- Explainer ----------------
def render_parameter_explainer():
    st.markdown("---")
    st.markdown("## Parameter explainer (plain English with TSLA examples)")

    st.markdown(
        """
**How to read this:** each knob says *what it does*, *when to nudge it up/down*, and a **TSLA-flavored example** so it’s easy to picture.
        """
    )

    st.markdown("### General")
    st.markdown(
        """
- **`n_components` (HMM states)**  
  How many hidden “modes” the model may learn (e.g., calm-bull, high-vol-bull, crash-bear, chop).  
  **Raise** if one state mixes very different behavior. **Lower** if states look redundant/unstable.  
  **TSLA example:** 3–4 often separates “bull-high-vol” from “true drawdown bear”.

- **`ema_span` (smoothing of bear probability)**  
  Smooths the daily bear-probability line. **Higher** = steadier (fewer whipsaws), **lower** = more sensitive.  
  **TSLA example:** 20 keeps most short noise out but still catches regime turns within weeks.

- **`k_forward` (look-ahead used only to name states)**  
  While training we label states by how the next *k* days behave on average.  
  **Bigger** = slower, trendier labeling; **smaller** = faster, noisier.  
  **TSLA example:** 7–10 works; 20 can be too slow for sharp 2022/2024 moves.
        """
    )

    st.markdown("### Hysteresis thresholds (when we enter/exit a bear candidate)")
    st.markdown(
        """
- **`bear_enter` (enter bear when smoothed bear-prob ≥ this)**  
  If the smoothed bear probability rises above this, we **start a bear candidate**.  
  **Higher** = fewer, stronger candidates.  
  **TSLA example:** 0.80 ≈ “don’t start a bear unless we’re ~80% sure.”

- **`bear_exit` (leave bear when ≤ this)**  
  We end bear once smoothed probability falls below this. Keep it **well below** `bear_enter` to avoid flip-flops.  
  **TSLA example:** enter 0.80 / exit 0.55.

- **`auto_thresholds` + `bear_target` + `auto_window_years` + `min_gap` (optional)**  
  Auto-pick `bear_enter`/`bear_exit` so that, over the last N years, bear time ≈ **`bear_target`** (e.g., 30%).  
  `min_gap` enforces distance between enter vs exit.  
  **Use when:** you want thresholds to adapt to the stock’s recent personality.
        """
    )

    st.markdown("### Bear confirmations (candidate → confirmed)")
    st.markdown(
        """
We **only** confirm a bear candidate if real weakness persists:

- **`mom_threshold` (trend weakness)**  
  Requires EMA20 to sit **below** EMA100 by at least this fraction of price.  
  **Higher** = stricter; **lower** = easier to confirm.  
  **TSLA example:** 0.03 ≈ EMA20 under EMA100 by ~3%.

- **`ddown_threshold` (drawdown from a recent peak)**  
  Price must be at least this far below a rolling peak (e.g., 15%).  
  **Higher** = confirm only deeper selloffs.  
  **TSLA example:** 0.15 catches clear slides like 2022.

- **`confirm_days` (persistence)**  
  How many days those conditions must hold. **Higher** = safer but later; **lower** = quicker but noisier.  
  **TSLA example:** 7 keeps one-week squiggles from “printing” official bears.

- **`min_bear_run` (cleanup)**  
  After confirming, drop tiny islands shorter than this many days so the picture looks sane.  
  **TSLA example:** 15 days removes blink-and-you-miss-it patches.
        """
    )

    st.markdown("### Bull confirmations & early exits")
    st.markdown(
        """
- **`bull_mom_threshold` (trend strength)**  
  Mirror of `mom_threshold`: EMA20 **above** EMA100 by at least this fraction to back a bull pocket.  
  **TSLA example:** 0.01 ensures rallies have some trend behind them.

- **`bull_ddown_exit` (drawdown recovered)**  
  If drawdown improves to within this distance of the prior peak (e.g., 6%), we **exit bear** even if probability lags.  
  **Use when:** “price fixed itself” should override a sticky probability.

- **`confirm_days_bull`**  
  Require bullish conditions for this many days before declaring the pocket bull.  
  **TSLA example:** 2–3 is fine.

- **`min_bull_run` (cleanup)**  
  Remove micro bull blips shorter than this many days inside a larger bear.

- **`bear_profit_exit` (bounce-off-entry fail-safe)**  
  If price jumps +X% from the **bear entry price**, exit the bear early.  
  **TSLA example:** 0.05 = “bounce 5% from entry → don’t keep calling it bear.”
        """
    )

    st.markdown("### Gates (filters at bear entry)")
    st.markdown(
        """
- **`direction_gate` (recent tape must be weak)**  
  Only allow bear entry if **both**:  
  • Last `entry_ret_lookback`-day return ≤ `entry_ret_thresh` (negative)  
  • Current drawdown ≤ `entry_dd_thresh` (negative)  
  **Why:** prevents starting a “bear” while price is actually rising.  
  **TSLA example:** L=10, ret ≤ −1%, drawdown ≤ −3%.

- **`trend_gate`**  
  Only allow bear entry if price is **under** the slow trend (EMA100).

- **`trend_exit_cross`**  
  If price **crosses back above** EMA100 while in bear, exit the bear even if other tests are slow to catch up.
        """
    )

    st.markdown("### Figure/UX")
    st.markdown(
        """
- **`zoom_years`**  
  How many years to show in the zoomed charts (e.g., 3). The full-history chart always shows IPO→today.
        """
    )

    st.markdown("### Candidate vs Confirmed (TSLA mini-example)")
    st.markdown(
        """
- **Candidate bear** = “probability alarm tripped.”  
  *Example:* After five rough TSLA days, the smoothed bear probability jumps above **0.80**. We **shade light red** (candidate). No regime switch yet.

- **Confirmed bear** = “the weakness is real and persistent.”  
  *Example:* Over the next week, EMA20 stays ≥3% **below** EMA100 and price stays ≥15% **under** the recent peak. We confirm bear (shade **dark red**).  
  If TSLA then **bounces +5%** from the entry price or **recovers near the peak** (within 6%), we **exit** the bear early.

> **Why you may still see a rising price inside a bear block**  
> The label reflects **ongoing risk conditions** (under-trend, deep drawdown), not just a single-day move. Short counter-trend pops can occur inside a broader down-move; they won’t flip a confirmed bear unless they persist or trip the fast-exit rules.
        """
    )

    st.markdown("### Quick tuning cheatsheet")
    st.markdown(
        """
- Too many false bears?  
  **Raise** `bear_enter`, `mom_threshold`, `ddown_threshold`, `confirm_days`; **enable** `direction_gate`.

- Bear exits too late?  
  **Lower** `bear_exit` slightly or **raise** `bull_ddown_exit` / **use** `trend_exit_cross` / **set** `bear_profit_exit`.

- Missing small bull pockets inside a big selloff?  
  **Lower** `bull_mom_threshold` and `confirm_days_bull`; reduce `min_bull_run`.

- Stock has a new personality (NVDA-style regime shift)?  
  Turn on **`auto_thresholds`** with a 3–5-year window and a sensible **`bear_target`** (e.g., 25–35%).
        """
    )

# Example usage:
render_parameter_explainer()
st.caption("Author: Dr. Poulami Nandi · Research demo only. Not investment advice.")
