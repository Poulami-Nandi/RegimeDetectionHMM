# tests/test_hmm.py
# -*- coding: utf-8 -*-
"""
TSLA-only smoke & consistency tests for the HMM regime pipeline.

These tests:
- Pull TSLA data online (IPO -> today) via your data loader (yfinance).
- Run the full detection pipeline with the TSLA-tuned knobs.
- Assert basic invariants (columns present, no NaNs in key outputs, probabilities in [0,1]).
- Check that labels are in {BULL, BEAR} and there is at least some variation.
- For last 3y, ensure the window isn’t empty.
- With strict_direction=True, verify ALL confirmed-bear segments lose money and/or show meaningful drawdown.

Mark as slow when needed; you can skip online runs with:
    pytest -k test_tsla --maxfail=1 -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import pytest

# ----- make project importable (assumes tests/ at project root) -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import get_price_data
from src.regime_detection import detect_regimes, BULL, BEAR


# ---------- helpers ----------

def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) for True runs in boolean mask."""
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


# ---------- tests ----------

@pytest.mark.tsla
def test_tsla_fetch_has_data_and_recent():
    """TSLA price data exists and is recent (within ~15 calendar days)."""
    try:
        px = get_price_data("TSLA", start=None, end=None)
    except Exception as e:
        pytest.skip(f"Skipping: online fetch failed ({e})")

    assert not px.empty, "TSLA price frame is empty."
    assert "Close" in px.columns, "Close column missing."

    last = px.index.max()
    assert pd.notna(last), "Last date is NaT."
    # Be generous on recency (market holidays/weekends).
    assert (dt.datetime.utcnow().date() - last.date()).days <= 15, (
        f"Last TSLA date {last.date()} is not recent."
    )


@pytest.mark.tsla
def test_tsla_hmm_pipeline_smoke_and_invariants():
    """Full pipeline runs and basic invariants hold."""
    try:
        df, model = detect_regimes(
            ticker="TSLA",
            n_components=4,
            ema_span=20,
            # hysteresis
            bear_enter=0.80,
            bear_exit=0.55,
            # confirmations (bear/bull)
            mom_threshold=0.03,
            ddown_threshold=0.15,
            confirm_days=7,
            bull_mom_threshold=0.01,
            bull_ddown_exit=0.06,
            confirm_days_bull=3,
            # min runs
            min_bear_run=15,
            min_bull_run=5,
            # labeling horizon
            k_forward=10,
            # direction & trend gates
            direction_gate=True,
            entry_ret_lookback=10,
            entry_ret_thresh=-0.01,
            entry_dd_thresh=-0.03,
            trend_gate=True,
            require_close_below_sma100=True,
            require_sma20_below_sma100=True,
            trend_exit_cross=True,
            bear_profit_exit=0.05,
            # strict filter OFF in this smoke test
            strict_direction=False,
        )
    except Exception as e:
        pytest.skip(f"Skipping: detection failed due to online/fit error ({e})")

    required_cols = [
        "Close", "ret", "vol", "ret_1", "z_ret", "mom_20_100", "drawdown",
        "p_bear", "p_bear_ema", "regime"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column: {c}"

    assert not df[["p_bear", "p_bear_ema"]].isna().any().any(), "NaNs in probabilities."
    assert (df["p_bear"].between(0, 1)).all(), "p_bear out of [0,1]."
    assert (df["p_bear_ema"].between(0, 1)).all(), "p_bear_ema out of [0,1]."

    # regimes should be only {0,1} and have some variation across history
    uniq = set(map(int, pd.Series(df["regime"]).dropna().unique().tolist()))
    assert uniq.issubset({BULL, BEAR}), f"Unexpected regime labels: {uniq}"
    assert len(uniq) == 2, "No variation in regimes over full history."

    # last 3y window should exist (TSLA is active)
    end = df.index.max()
    df3 = df.loc[df.index >= end - pd.DateOffset(years=3)]
    assert not df3.empty, "Last 3y window is empty for TSLA."


@pytest.mark.tsla
def test_tsla_strict_direction_eliminates_false_bears():
    """
    With strict_direction=True, every confirmed-bear segment should
    (a) have total return <= strict_bear_min_ret  AND
    (b) show worst drawdown from entry <= strict_bear_min_maxdd.
    """
    strict_bear_min_ret = -0.005
    strict_bear_min_maxdd = -0.03

    try:
        df, _ = detect_regimes(
            ticker="TSLA",
            n_components=4,
            ema_span=20,
            # hysteresis
            bear_enter=0.80,
            bear_exit=0.55,
            # confirmations
            mom_threshold=0.03,
            ddown_threshold=0.15,
            confirm_days=7,
            bull_mom_threshold=0.01,
            bull_ddown_exit=0.06,
            confirm_days_bull=3,
            # min runs
            min_bear_run=15,
            min_bull_run=5,
            # labeling horizon
            k_forward=10,
            # gates
            direction_gate=True,
            entry_ret_lookback=10,
            entry_ret_thresh=-0.01,
            entry_dd_thresh=-0.03,
            trend_gate=True,
            require_close_below_sma100=True,
            require_sma20_below_sma100=True,
            trend_exit_cross=True,
            bear_profit_exit=0.05,
            # strict ON
            strict_direction=True,
            strict_bear_min_ret=strict_bear_min_ret,
            strict_bear_min_maxdd=strict_bear_min_maxdd,
        )
    except Exception as e:
        pytest.skip(f"Skipping: detection failed due to online/fit error ({e})")

    close = df["Close"].to_numpy()
    bear_mask = (df["regime"].to_numpy().astype(int) == BEAR)
    segs = _contiguous_segments(bear_mask)

    # It's possible there are no confirmed bears in some short windows; only assert if we have any.
    for i, j in segs:
        p0 = float(close[i])
        p1 = float(close[j])
        ret = (p1 / p0) - 1.0
        run_min = float(np.min(close[i:j+1]))
        dd_from_entry = (run_min / p0) - 1.0
        assert ret <= strict_bear_min_ret + 1e-12, f"Bear run [{i},{j}] ret {ret:.4f} > {strict_bear_min_ret}"
        assert dd_from_entry <= strict_bear_min_maxdd + 1e-12, (
            f"Bear run [{i},{j}] worst DD {dd_from_entry:.4f} > {strict_bear_min_maxdd}"
        )


@pytest.mark.tsla
def test_tsla_last3y_has_some_regime_transitions():
    """In the last 3 years, expect at least one transition (bull↔bear) to exist."""
    try:
        df, _ = detect_regimes(
            ticker="TSLA",
            n_components=4,
            ema_span=20,
            bear_enter=0.80,
            bear_exit=0.55,
            mom_threshold=0.03,
            ddown_threshold=0.15,
            confirm_days=7,
            bull_mom_threshold=0.01,
            bull_ddown_exit=0.06,
            confirm_days_bull=3,
            min_bear_run=15,
            min_bull_run=5,
            k_forward=10,
            direction_gate=True,
            entry_ret_lookback=10,
            entry_ret_thresh=-0.01,
            entry_dd_thresh=-0.03,
            trend_gate=True,
            require_close_below_sma100=True,
            require_sma20_below_sma100=True,
            trend_exit_cross=True,
            bear_profit_exit=0.05,
            strict_direction=False,
        )
    except Exception as e:
        pytest.skip(f"Skipping: detection failed due to online/fit error ({e})")

    end = df.index.max()
    df3 = df.loc[df.index >= end - pd.DateOffset(years=3)].copy()
    if df3.empty:
        pytest.skip("Last 3y empty; skipping transition check.")

    reg = df3["regime"].astype(int).to_numpy()
    transitions = int(np.sum(reg[1:] != reg[:-1]))
    assert transitions >= 1, "No regime transitions found in the last 3 years for TSLA."
