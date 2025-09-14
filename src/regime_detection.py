from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .data_loader import get_price_data
from .features import make_features
from .hmm_model import RegimeHMM

# We standardize on 2 canonical *macro* labels:
#   BULL = 0, BEAR = 1
# The HMM internally may have many latent states; later we map those
# to these two buckets (bull/bear) using a composite “bearishness” score.
BULL = 0
BEAR = 1


# ---------- small helpers (stateless, side-effect free) ----------

def _ema(s: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average (EMA) with no ‘adjust’ backfill.
    Used to smooth posterior bear probabilities before thresholding
    so the candidate/confirmed logic is less twitchy on noisy days.
    """
    return s.ewm(span=span, adjust=False).mean()


def _k_forward_returns(rets: pd.Series, k: int) -> pd.Series:
    """
    Forward sum of (log) returns from t+1..t+k.
    Intuition: a state is “bearish” if, on average, the *future* k-day
    return after being in that state tends to be negative.

    We use this only for *naming* states during training – not for trading.
    That keeps the model honest while still giving human-meaningful labels.
    """
    return rets.shift(-1).rolling(k, min_periods=1).sum()


def _composite_state_scores(states: np.ndarray, feats: pd.DataFrame, k_forward: int) -> Dict[int, float]:
    """
    Compute a *bearishness* score for every latent HMM state.

    Why: The HMM emits unlabeled states (0..K-1). We want to call some of
    them “bearish” and others “bullish” without hand-picking. So we score
    each state using features that capture *future loss* and *downside risk*.

    Score definition (higher → more bear):
      - negative forward k-day return (weight 1.0)
      - drawdown depth (0.5)
      - volatility (0.05)   <-- tiny weight: being “up on high vol” isn’t bear
      - fraction of time with SMA20 < SMA100 (0.5) → persistent downtrend

    Parameters
    ----------
    states : array[int]
        Viterbi path from the HMM (state index per day).
    feats : DataFrame
        Feature matrix returned by `make_features` (must include 'ret',
        'drawdown', 'vol', 'mom_20_100').
    k_forward : int
        Look-ahead horizon for forward return labeling (naming only).

    Returns
    -------
    Dict[int, float]
        state_id → bearishness score
    """
    fwdk = _k_forward_returns(feats["ret"], k_forward)
    scores: Dict[int, float] = {}
    for s in np.unique(states):
        m = (states == s)
        mean_fwd_k = fwdk[m].mean()                              # future return (want negative for bears)
        mean_dd = np.maximum(0.0, -feats.loc[m, "drawdown"]).mean()  # convert ≤0 drawdown to positive depth
        mean_vol = feats.loc[m, "vol"].mean()                    # volatility proxy
        frac_trendneg = (feats.loc[m, "mom_20_100"] < 0).mean()  # how often fast MA < slow MA

        # Be careful with NaNs to avoid poisoning the score
        score = (-(0.0 if pd.isna(mean_fwd_k) else mean_fwd_k)) \
              + 0.5*(0.0 if pd.isna(mean_dd) else mean_dd) \
              + 0.05*(0.0 if pd.isna(mean_vol) else mean_vol) \
              + 0.5*(0.0 if pd.isna(frac_trendneg) else frac_trendneg)
        scores[int(s)] = float(score)
    return scores


def _assign_bull_bear(scores: Dict[int, float]):
    """
    Split HMM states into BULL vs BEAR using the median score.

    Why median? It’s robust and avoids hand-tuned cutoffs. If scores are
    very skewed, at least half the mass sits on each side.

    Returns
    -------
    mapping : dict[int, int]
        state_id → BULL(0)/BEAR(1)
    bull_states : list[int]
        State ids mapped to BULL for convenience.
    bear_states : list[int]
        State ids mapped to BEAR for convenience (used to sum posterior).
    """
    st = sorted(scores.items(), key=lambda kv: kv[1])
    median = float(np.median([v for _, v in st]))

    # BEAR = strictly above the median; if all tie, at least the max one is bear.
    bear_states = [s for s, v in scores.items() if v > median] or [st[-1][0]]
    bull_states = [s for s in scores if s not in bear_states] or [st[0][0]]

    mapping = {s: (BEAR if s in bear_states else BULL) for s in scores.keys()}
    return mapping, bull_states, bear_states


def _consecutive_prune(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Prune short True-islands (run length < min_len) in a boolean mask.

    Used to require *persistence* in confirm conditions. Example:
    you may require 7 consecutive days of trend weakness before
    declaring a “confirmed bear”.
    """
    if min_len <= 1 or mask.size == 0: 
        return mask
    out = mask.copy(); i = 0; n = len(out)
    while i < n:
        if not out[i]: 
            i += 1; continue
        j = i
        while j + 1 < n and out[j+1]: 
            j += 1
        if (j - i + 1) < min_len: 
            out[i:j+1] = False
        i = j + 1
    return out


def _clean_islands_protected(labels: np.ndarray, min_bull: int, min_bear: int, protect_bull: np.ndarray | None = None) -> np.ndarray:
    """
    Remove tiny regime islands after the main pathing, but NEVER flip any index
    where protect_bull[idx] is True.

    Why: we sometimes mark “protected bull” points when exiting bear early
    due to a bounce (profit-exit). Tweaking cleaners later should not erase
    those explicit exit decisions.

    Parameters
    ----------
    labels : array[int]
        Sequence of BULL/BEAR labels per day.
    min_bull : int
        Minimum allowed run length for a bull island.
    min_bear : int
        Minimum allowed run length for a bear island.
    protect_bull : array[bool] | None
        If provided, any True positions are immune to flipping to BEAR.

    Returns
    -------
    array[int]
        Cleaned labels, honoring protection.
    """
    y = labels.copy()
    n = len(y)
    if protect_bull is None:
        protect_bull = np.zeros(n, dtype=bool)

    def _rewrite(i, j, value_to_set):
        # If we try to flip to BEAR but any element is protected bull, skip.
        if value_to_set == BEAR and protect_bull[i:j+1].any():
            return
        y[i:j+1] = value_to_set

    # prune short BEAR runs
    i = 0
    while i < n:
        if y[i] != BEAR: 
            i += 1; continue
        j = i
        while j+1 < n and y[j+1] == BEAR: 
            j += 1
        if (j - i + 1) < min_bear:
            left = y[i-1] if i > 0 else None
            right = y[j+1] if j+1 < n else None
            _rewrite(i, j, right if right is not None else (left if left is not None else y[i]))
        i = j + 1

    # prune short BULL runs (respect protect_bull)
    i = 0
    while i < n:
        if y[i] != BULL: 
            i += 1; continue
        j = i
        while j+1 < n and y[j+1] == BULL: 
            j += 1
        if (j - i + 1) < min_bull and not protect_bull[i:j+1].any():
            left = y[i-1] if i > 0 else None
            right = y[j+1] if j+1 < n else None
            y[i:j+1] = right if right is not None else (left if left is not None else y[i])
        i = j + 1

    return y


def _hysteresis_path_directional(
    p_bear_ema: np.ndarray,
    close: np.ndarray,
    trail_ret: np.ndarray,
    drawdown: np.ndarray,
    sma20: np.ndarray,
    sma100: np.ndarray,
    enter: float,
    exit_: float,
    bear_c: np.ndarray,
    bull_c: np.ndarray,
    use_direction_gate: bool,
    entry_ret_thresh: float,
    entry_dd_thresh: float,
    trend_gate: bool,
    require_close_below_sma100: bool,
    require_sma20_below_sma100: bool,
    trend_exit_cross: bool,
    bear_profit_exit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the final bull/bear path using *hysteresis* and *gates*:

    Hysteresis:
      - Enter bear when smoothed bear prob >= enter (and confirmations pass).
      - Exit bear when smoothed bear prob <= exit_ (or a bull confirmation / bounce / trend cross fires).
      Using two thresholds prevents rapid flip-flopping.

    Direction gate (optional):
      - Only enter bear if recent return is weak (trail_ret <= entry_ret_thresh)
        AND drawdown is meaningful (drawdown <= entry_dd_thresh).

    Trend gate (optional):
      - Only enter bear if Close < SMA100 and/or SMA20 < SMA100.
      - Optional trend_exit_cross: if price/MA relationships flip during bear, exit early.

    Early profit-exit from bear (optional):
      - If the price has bounced up from the bear-entry price by ≥ bear_profit_exit,
        mark a bull point (protected) and exit bear even if probability hasn’t yet dropped.

    Returns
    -------
    y : array[int]
        Sequence of BULL/BEAR labels.
    protect_bull : array[bool]
        Mask of “protected bull” indices (used to safeguard from later cleaners).
    """
    n = len(p_bear_ema)
    y = np.zeros(n, dtype=int)
    protect_bull = np.zeros(n, dtype=bool)
    in_bear = False
    entry_price = np.nan

    for t in range(n):
        if not in_bear:
            # Eligible to *enter* bear if prob high and (optionally) gates pass
            ok_prob = (p_bear_ema[t] >= enter) and bear_c[t]

            ok_dir = True
            if use_direction_gate:
                ok_tr = (not np.isnan(trail_ret[t])) and (trail_ret[t] <= entry_ret_thresh)
                ok_dd = (not np.isnan(drawdown[t])) and (drawdown[t] <= entry_dd_thresh)
                ok_dir = ok_tr and ok_dd

            ok_trend = True
            if trend_gate:
                conds = []
                if require_close_below_sma100:
                    conds.append(close[t] < sma100[t])
                if require_sma20_below_sma100:
                    conds.append(sma20[t] < sma100[t])
                ok_trend = all(conds) if conds else True

            if ok_prob and ok_dir and ok_trend:
                in_bear = True
                entry_price = close[t]

        else:
            # Conditions to *exit* bear
            exit_prob = (p_bear_ema[t] <= exit_) or bull_c[t]

            exit_bounce = False
            if use_direction_gate and entry_price > 0:
                # bounce % since entry; if large enough, profit-exit the bear
                since = (close[t] / entry_price) - 1.0
                exit_bounce = (since >= bear_profit_exit)

            exit_trend = False
            if trend_exit_cross:
                # If the same trend gates that required “below” have now failed, exit
                failed_close = (close[t] >= sma100[t]) if require_close_below_sma100 else False
                failed_sma   = (sma20[t] >= sma100[t]) if require_sma20_below_sma100 else False
                exit_trend = failed_close or failed_sma

            if exit_prob or exit_bounce or exit_trend:
                in_bear = False
                entry_price = np.nan
                protect_bull[t] = True  # protect this bull flip from cleaners

        y[t] = BEAR if in_bear else BULL

    return y, protect_bull


# ---------- high-level API (used by the app) ----------

def detect_regimes(
    ticker: str = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    n_components: int = 3,
    ema_span: int = 12,

    # fixed thresholds for hysteresis
    bear_enter: float = 0.75,
    bear_exit: float = 0.55,

    # auto thresholds (derive enter/exit from the data distribution)
    auto_thresholds: bool = False,
    enter_quantile: float = 0.75,
    exit_quantile: float = 0.55,
    bear_target: Optional[float] = None,
    auto_window_years: Optional[int] = 5,
    min_gap: float = 0.10,
    min_spread: float = 0.05,
    std_floor: float = 5e-3,

    # confirmations (trend & drawdown)
    mom_threshold: float = 0.02,
    ddown_threshold: float = 0.10,
    confirm_days: int = 5,
    bull_mom_threshold: float = 0.00,
    bull_ddown_exit: float = 0.06,
    confirm_days_bull: int = 3,

    # minimum run-lengths (clean tiny islands)
    min_bear_run: int = 12,
    min_bull_run: int = 5,

    # horizon used only for naming states (not trading)
    k_forward: int = 5,

    # directional + trend gates around the hysteresis
    direction_gate: bool = True,
    entry_ret_lookback: int = 10,
    entry_ret_thresh: float = -0.01,
    entry_dd_thresh: float = -0.03,
    trend_gate: bool = True,
    require_close_below_sma100: bool = True,
    require_sma20_below_sma100: bool = True,
    trend_exit_cross: bool = True,
    bear_profit_exit: float = 0.05,

    # an optional stricter pass that throws away unproductive bear segments
    strict_direction: bool = False,
    strict_bear_min_ret: float = -0.005,
    strict_bear_min_maxdd: float = -0.03,
):
    """
    Full pipeline:
      1) Load prices and build features (vol, drawdown, MA gaps, returns, etc.)
      2) Fit a stabilized HMM (KMeans init, diag covars, restarts)
      3) Map latent states to bull/bear via composite “bearishness” scores
      4) Smooth posterior bear prob with EMA
      5) Build confirm signals (trend & drawdown) with persistence
      6) Choose enter/exit thresholds (fixed or auto-calibrated)
      7) Apply hysteresis with direction/trend gates to get the path
      8) Optionally drop bear runs that don’t actually lose or draw down (strict)
      9) Clean tiny islands (with protection for bounce exits)
     10) Assemble an output DataFrame with diagnostics and probabilities

    Returns
    -------
    out : pd.DataFrame
        Indexed by date with columns:
          - Close, state_raw (HMM), regime_viterbi (BULL/BEAR mapped),
            p_s*, p_bear, p_bear_ema, *_confirm*, bear_confirm, bull_confirm, regime
    model : RegimeHMM
        The trained HMM, with `.thresholds_` attribute storing chosen enter/exit.
    """
    # 1) data & features (features supply drawdown, vol, momentum gap, returns, etc.)
    price = get_price_data(ticker, start, end)
    feats = make_features(price).copy()
    close_series = price.loc[feats.index, "Close"]
    close = close_series.values

    # SMAs used by the trend gates (not the same as mom_20_100, which is a ratio/gap feature)
    sma20_series = close_series.rolling(20, min_periods=5).mean()
    sma100_series = close_series.rolling(100, min_periods=20).mean()
    sma20 = sma20_series.values
    sma100 = sma100_series.values

    dd = feats["drawdown"].values  # features drawdown is ≤ 0 (0 at peaks, negative below peaks)

    # 2) HMM (stabilized with standardization, diag covars, restarts)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(feats.values)
    model = RegimeHMM(
        n_components=n_components, n_iter=250, tol=1e-3,
        covariance_type="diag", n_restarts=3, random_state=42
    ).fit(X)
    states = model.predict(X)   # hard labels (Viterbi)
    post = model.posterior(X)   # soft labels (posterior probs)

    # 3) map latent states → bull/bear via composite score
    scores = _composite_state_scores(states, feats, k_forward=k_forward)
    mapping, bull_states, bear_states = _assign_bull_bear(scores)
    regime_vit = np.vectorize(lambda s: mapping.get(s, s))(states)

    # 4) posterior bear prob and smooth it with EMA for stability
    p_bear = post[:, bear_states].sum(axis=1) if len(bear_states) > 1 else post[:, bear_states[0]]
    p_bear = pd.Series(p_bear, index=feats.index, name="p_bear")
    p_bear_ema = _ema(p_bear, span=ema_span).rename("p_bear_ema")

    # 5) confirmations: trend & drawdown, with persistence pruning
    #    (bear confirms if either trend is weak OR drawdown deep; then require N consecutive days)
    bear_c_tr = (feats["mom_20_100"] < -abs(mom_threshold)).values
    bear_c_dd = (feats["drawdown"] < -abs(ddown_threshold)).values
    bear_c = _consecutive_prune(np.logical_or(bear_c_tr, bear_c_dd), confirm_days)

    #    (bull confirms if trend is positive OR drawdown has healed; then require N consecutive days)
    bull_c_tr = (feats["mom_20_100"] > abs(bull_mom_threshold)).values
    bull_c_dd = (feats["drawdown"] > -abs(bull_ddown_exit)).values
    bull_c = _consecutive_prune(np.logical_or(bull_c_tr, bull_c_dd), confirm_days_bull)

    # 6) thresholds (fixed or auto): choose enter/exit on the smoothed p_bear
    used_enter, used_exit = float(bear_enter), float(bear_exit)
    if auto_thresholds:
        series = p_bear_ema.copy()
        if auto_window_years and auto_window_years > 0:
            end_t = series.index.max()
            series = series.loc[series.index >= end_t - pd.DateOffset(years=auto_window_years)]
        pb = series.dropna().values

        # Option A: match a target bear fraction over the window
        if bear_target is not None:
            qE = min(0.98, 1.0 - bear_target/2.0)   # higher quantile → rarer entries
            qX = min(0.95, 1.0 - bear_target)       # lower quantile → earlier exits
        else:
            qE, qX = float(enter_quantile), float(exit_quantile)

        # Guard-rails: keep a sensible spread and avoid degenerate cases
        qE = max(0.55, min(0.99, qE))
        qX = max(0.05, min(qE - 0.02, qX))

        if (len(pb) < 50) or (np.nanstd(pb) < std_floor):
            # If the distribution is too thin, fall back to conservative defaults
            used_enter, used_exit = 0.72, 0.50
        else:
            qe = np.nanquantile(pb, qE); qx = np.nanquantile(pb, qX)
            if (qe - qx) < min_spread:
                used_enter, used_exit = max(qe, 0.60), max(min(qx, qe - min_gap), 0.30)
            else:
                used_enter, used_exit = float(qe), float(qx)

        used_enter = float(np.clip(used_enter, 0.10, 0.95))
        used_exit  = float(np.clip(used_exit, 0.05, used_enter - min_gap))

    # 7) directional context for the entry gate (rolling sum of returns)
    trail_ret = feats["ret"].rolling(entry_ret_lookback, min_periods=entry_ret_lookback).sum().values

    # 8) build the hysteresis path (with direction/trend gates + profit-exit)
    regime, protect_bull = _hysteresis_path_directional(
        p_bear_ema.values, close, trail_ret, dd, sma20, sma100,
        enter=used_enter, exit_=used_exit,
        bear_c=bear_c, bull_c=bull_c,
        use_direction_gate=direction_gate,
        entry_ret_thresh=entry_ret_thresh,
        entry_dd_thresh=entry_dd_thresh,
        trend_gate=trend_gate,
        require_close_below_sma100=require_close_below_sma100,
        require_sma20_below_sma100=require_sma20_below_sma100,
        trend_exit_cross=trend_exit_cross,
        bear_profit_exit=bear_profit_exit,
    )

    # 9) strict pass (optional): if a bear run didn’t actually lose or draw down,
    #    demote it to bull. This reduces false alarms in noisy sideways markets.
    if strict_direction:
        y = regime.copy()
        n = len(y); i = 0
        while i < n:
            if y[i] != BEAR:
                i += 1; continue
            j = i
            while j + 1 < n and y[j+1] == BEAR: 
                j += 1
            p0 = close[i]; p1 = close[j]
            ret = (p1 / p0) - 1.0
            run_min = np.min(close[i:j+1]) if j >= i else p1
            dd_from_entry = (run_min / p0) - 1.0
            if (ret > strict_bear_min_ret) or (dd_from_entry > strict_bear_min_maxdd):
                # Not bearish enough → flip to bull and protect from cleaners
                y[i:j+1] = BULL
                protect_bull[i:j+1] = True
            i = j + 1
        regime = y

    # 10) after flips are decided, remove tiny islands; keep protected bulls intact
    regime = _clean_islands_protected(regime, min_bull=min_bull_run, min_bear=min_bear_run, protect_bull=protect_bull)

    # 11) Package outputs with rich diagnostics for plotting / analysis
    out = feats.copy()
    out["state_raw"] = states              # HMM latent state id (0..K-1)
    out["regime_viterbi"] = regime_vit     # latent mapped to macro BULL/BEAR by score
    out["Close"] = close_series

    # posterior by state (handy when debugging)
    for si in range(n_components):
        out[f"p_s{si}"] = post[:, si]

    # bear probability (sum of bear states) + smoothed version
    out["p_bear"] = p_bear
    out["p_bear_ema"] = _ema(p_bear, span=ema_span)

    # raw confirm components (before pruning)
    out["bear_confirm_trend"] = (feats["mom_20_100"] < -abs(mom_threshold)).values
    out["bear_confirm_dd"] = (feats["drawdown"] < -abs(ddown_threshold)).values
    out["bull_confirm_trend"] = (feats["mom_20_100"] > abs(bull_mom_threshold)).values
    out["bull_confirm_dd"] = (feats["drawdown"] > -abs(bull_ddown_exit)).values

    # pruned confirmations used in the path
    out["bear_confirm"] = _consecutive_prune(np.logical_or(out["bear_confirm_trend"], out["bear_confirm_dd"]), confirm_days)
    out["bull_confirm"] = _consecutive_prune(np.logical_or(out["bull_confirm_trend"], out["bull_confirm_dd"]), confirm_days_bull)

    # final path
    out["regime"] = regime

    # stash chosen thresholds on the model so the UI can display them exactly
    setattr(model, "thresholds_", {"bear_enter": used_enter, "bear_exit": used_exit})
    return out, model
