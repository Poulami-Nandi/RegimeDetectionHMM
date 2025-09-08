from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .data_loader import get_price_data
from .features import make_features
from .hmm_model import RegimeHMM

BULL = 0
BEAR = 1

# ---------- helpers ----------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _k_forward_returns(rets: pd.Series, k: int) -> pd.Series:
    # forward sum of (log) returns from t+1..t+k
    return rets.shift(-1).rolling(k, min_periods=1).sum()

def _composite_state_scores(states: np.ndarray, feats: pd.DataFrame, k_forward: int) -> Dict[int, float]:
    """
    Bearishness score (higher = more bear):
      - negative forward k-day return (weight 1.0)
      - drawdown depth (0.5)
      - volatility (0.05)   <-- low weight to avoid "up on high vol" = bear
      - % time SMA20<SMA100 (0.5)
    """
    fwdk = _k_forward_returns(feats["ret"], k_forward)
    scores: Dict[int, float] = {}
    for s in np.unique(states):
        m = (states == s)
        mean_fwd_k = fwdk[m].mean()
        mean_dd = np.maximum(0.0, -feats.loc[m, "drawdown"]).mean()
        mean_vol = feats.loc[m, "vol"].mean()
        frac_trendneg = (feats.loc[m, "mom_20_100"] < 0).mean()
        score = (-(0.0 if pd.isna(mean_fwd_k) else mean_fwd_k)) \
              + 0.5*(0.0 if pd.isna(mean_dd) else mean_dd) \
              + 0.05*(0.0 if pd.isna(mean_vol) else mean_vol) \
              + 0.5*(0.0 if pd.isna(frac_trendneg) else frac_trendneg)
        scores[int(s)] = float(score)
    return scores

def _assign_bull_bear(scores: Dict[int, float]):
    st = sorted(scores.items(), key=lambda kv: kv[1])
    median = float(np.median([v for _, v in st]))
    bear_states = [s for s, v in scores.items() if v > median] or [st[-1][0]]
    bull_states = [s for s in scores if s not in bear_states] or [st[0][0]]
    mapping = {s: (BEAR if s in bear_states else BULL) for s in scores.keys()}
    return mapping, bull_states, bear_states

def _consecutive_prune(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or mask.size == 0: return mask
    out = mask.copy(); i = 0; n = len(out)
    while i < n:
        if not out[i]: i += 1; continue
        j = i
        while j + 1 < n and out[j+1]: j += 1
        if (j - i + 1) < min_len: out[i:j+1] = False
        i = j + 1
    return out

def _clean_islands_protected(labels: np.ndarray, min_bull: int, min_bear: int, protect_bull: np.ndarray | None = None) -> np.ndarray:
    """
    Remove tiny runs of BEAR/BULL, but NEVER flip any index where protect_bull[idx] is True.
    This preserves bull points created by a bounce-exit, so cleaners cannot overwrite them.
    """
    y = labels.copy()
    n = len(y)
    if protect_bull is None:
        protect_bull = np.zeros(n, dtype=bool)

    def _rewrite(i, j, value_to_set):
        if value_to_set == BEAR and protect_bull[i:j+1].any():
            return
        y[i:j+1] = value_to_set

    # prune short BEAR runs
    i = 0
    while i < n:
        if y[i] != BEAR: i += 1; continue
        j = i
        while j+1 < n and y[j+1] == BEAR: j += 1
        if (j - i + 1) < min_bear:
            left = y[i-1] if i > 0 else None
            right = y[j+1] if j+1 < n else None
            _rewrite(i, j, right if right is not None else (left if left is not None else y[i]))
        i = j + 1

    # prune short BULL runs (respect protect_bull)
    i = 0
    while i < n:
        if y[i] != BULL: i += 1; continue
        j = i
        while j+1 < n and y[j+1] == BULL: j += 1
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
    Two-threshold hysteresis with:
      • Direction gate at entry: trail_ret <= thresh AND drawdown <= thresh
      • Trend gate at entry (optional): Close<SMA100 and/or SMA20<SMA100
      • Exit on: prob exit, bull confirms, bounce profit-exit, or trend cross (optional)
    Returns labels and a mask of bull points to protect from post cleaners.
    """
    n = len(p_bear_ema)
    y = np.zeros(n, dtype=int)
    protect_bull = np.zeros(n, dtype=bool)
    in_bear = False
    entry_price = np.nan

    for t in range(n):
        if not in_bear:
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
            exit_prob = (p_bear_ema[t] <= exit_) or bull_c[t]
            exit_bounce = False
            if use_direction_gate and entry_price > 0:
                since = (close[t] / entry_price) - 1.0
                exit_bounce = (since >= bear_profit_exit)
            exit_trend = False
            if trend_exit_cross:
                # exit if both trend gates are no longer satisfied
                failed_close = (close[t] >= sma100[t]) if require_close_below_sma100 else False
                failed_sma   = (sma20[t] >= sma100[t]) if require_sma20_below_sma100 else False
                exit_trend = failed_close or failed_sma
            if exit_prob or exit_bounce or exit_trend:
                in_bear = False
                entry_price = np.nan
                protect_bull[t] = True
        y[t] = BEAR if in_bear else BULL
    return y, protect_bull

# ---------- main ----------

def detect_regimes(
    ticker: str = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    n_components: int = 3,
    ema_span: int = 12,

    # fixed thresholds
    bear_enter: float = 0.75,
    bear_exit: float = 0.55,

    # auto thresholds
    auto_thresholds: bool = False,
    enter_quantile: float = 0.75,
    exit_quantile: float = 0.55,
    bear_target: Optional[float] = None,
    auto_window_years: Optional[int] = 5,
    min_gap: float = 0.10,
    min_spread: float = 0.05,
    std_floor: float = 5e-3,

    # confirmations
    mom_threshold: float = 0.02,
    ddown_threshold: float = 0.10,
    confirm_days: int = 5,
    bull_mom_threshold: float = 0.00,
    bull_ddown_exit: float = 0.06,
    confirm_days_bull: int = 3,

    # run-lengths
    min_bear_run: int = 12,
    min_bull_run: int = 5,

    # labeling horizon
    k_forward: int = 5,

    # directional + trend gates
    direction_gate: bool = True,
    entry_ret_lookback: int = 10,
    entry_ret_thresh: float = -0.01,
    entry_dd_thresh: float = -0.03,
    trend_gate: bool = True,
    require_close_below_sma100: bool = True,
    require_sma20_below_sma100: bool = True,
    trend_exit_cross: bool = True,
    bear_profit_exit: float = 0.05,

    # strict direction (optional)
    strict_direction: bool = False,
    strict_bear_min_ret: float = -0.005,
    strict_bear_min_maxdd: float = -0.03,
):
    # 1) data & features
    price = get_price_data(ticker, start, end)
    feats = make_features(price).copy()
    close_series = price.loc[feats.index, "Close"]
    close = close_series.values

    # SMAs for trend gate
    sma20_series = close_series.rolling(20, min_periods=5).mean()
    sma100_series = close_series.rolling(100, min_periods=20).mean()
    sma20 = sma20_series.values
    sma100 = sma100_series.values

    dd = feats["drawdown"].values  # (≤0)

    # 2) HMM (stabilized)
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(feats.values)
    model = RegimeHMM(n_components=n_components, n_iter=250, tol=1e-3,
                      covariance_type="diag", n_restarts=3, random_state=42).fit(X)
    states = model.predict(X)
    post = model.posterior(X)

    # 3) map states → bull/bear
    scores = _composite_state_scores(states, feats, k_forward=k_forward)
    mapping, bull_states, bear_states = _assign_bull_bear(scores)
    regime_vit = np.vectorize(lambda s: mapping.get(s, s))(states)

    # 4) probs & EMA
    p_bear = post[:, bear_states].sum(axis=1) if len(bear_states) > 1 else post[:, bear_states[0]]
    p_bear = pd.Series(p_bear, index=feats.index, name="p_bear")
    p_bear_ema = _ema(p_bear, span=ema_span).rename("p_bear_ema")

    # 5) confirmations
    bear_c_tr = (feats["mom_20_100"] < -abs(mom_threshold)).values
    bear_c_dd = (feats["drawdown"] < -abs(ddown_threshold)).values
    bear_c = _consecutive_prune(np.logical_or(bear_c_tr, bear_c_dd), confirm_days)

    bull_c_tr = (feats["mom_20_100"] > abs(bull_mom_threshold)).values
    bull_c_dd = (feats["drawdown"] > -abs(bull_ddown_exit)).values
    bull_c = _consecutive_prune(np.logical_or(bull_c_tr, bull_c_dd), confirm_days_bull)

    # 6) thresholds (auto/fixed)
    used_enter, used_exit = float(bear_enter), float(bear_exit)
    if auto_thresholds:
        series = p_bear_ema.copy()
        if auto_window_years and auto_window_years > 0:
            end_t = series.index.max()
            series = series.loc[series.index >= end_t - pd.DateOffset(years=auto_window_years)]
        pb = series.dropna().values
        if bear_target is not None:
            qE = min(0.98, 1.0 - bear_target/2.0)
            qX = min(0.95, 1.0 - bear_target)
        else:
            qE, qX = float(enter_quantile), float(exit_quantile)
        qE = max(0.55, min(0.99, qE))
        qX = max(0.05, min(qE - 0.02, qX))
        if (len(pb) < 50) or (np.nanstd(pb) < std_floor):
            used_enter, used_exit = 0.72, 0.50
        else:
            qe = np.nanquantile(pb, qE); qx = np.nanquantile(pb, qX)
            if (qe - qx) < min_spread:
                used_enter, used_exit = max(qe, 0.60), max(min(qx, qe - min_gap), 0.30)
            else:
                used_enter, used_exit = float(qe), float(qx)
        used_enter = float(np.clip(used_enter, 0.10, 0.95))
        used_exit  = float(np.clip(used_exit, 0.05, used_enter - min_gap))

    # 7) directional inputs
    trail_ret = feats["ret"].rolling(entry_ret_lookback, min_periods=entry_ret_lookback).sum().values

    # 8) hysteresis + direction + trend → labels + protected bull points
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

    # 9) STRICT direction (optional): drop bear runs that don't lose or lack DD
    if strict_direction:
        y = regime.copy()
        n = len(y); i = 0
        while i < n:
            if y[i] != BEAR:
                i += 1; continue
            j = i
            while j + 1 < n and y[j+1] == BEAR: j += 1
            p0 = close[i]; p1 = close[j]
            ret = (p1 / p0) - 1.0
            run_min = np.min(close[i:j+1]) if j >= i else p1
            dd_from_entry = (run_min / p0) - 1.0
            if (ret > strict_bear_min_ret) or (dd_from_entry > strict_bear_min_maxdd):
                y[i:j+1] = BULL
                protect_bull[i:j+1] = True
            i = j + 1
        regime = y

    # 10) remove tiny islands AFTER flips (respect protected bulls)
    regime = _clean_islands_protected(regime, min_bull=min_bull_run, min_bear=min_bear_run, protect_bull=protect_bull)

    # 11) assemble dataframe
    out = feats.copy()
    out["state_raw"] = states
    out["regime_viterbi"] = regime_vit
    out["Close"] = close_series
    for si in range(n_components):
        out[f"p_s{si}"] = post[:, si]
    out["p_bear"] = p_bear
    out["p_bear_ema"] = _ema(p_bear, span=ema_span)
    out["bear_confirm_trend"] = (feats["mom_20_100"] < -abs(mom_threshold)).values
    out["bear_confirm_dd"] = (feats["drawdown"] < -abs(ddown_threshold)).values
    out["bull_confirm_trend"] = (feats["mom_20_100"] > abs(bull_mom_threshold)).values
    out["bull_confirm_dd"] = (feats["drawdown"] > -abs(bull_ddown_exit)).values
    out["bear_confirm"] = _consecutive_prune(np.logical_or(out["bear_confirm_trend"], out["bear_confirm_dd"]), confirm_days)
    out["bull_confirm"] = _consecutive_prune(np.logical_or(out["bull_confirm_trend"], out["bull_confirm_dd"]), confirm_days_bull)
    out["regime"] = regime

    setattr(model, "thresholds_", {"bear_enter": used_enter, "bear_exit": used_exit})
    return out, model
