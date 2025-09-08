from __future__ import annotations
import pandas as pd
from pathlib import Path
from .utils import get_logger

log = get_logger("data")


def _read_local_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["Close"]]


def _synthetic_data(n: int = 800) -> pd.DataFrame:
    import numpy as np
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    np.random.seed(42)
    rets = 0.0003 + 0.01 * np.random.randn(n)
    price = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Close": price}, index=idx)


def _fallback_local_or_synth() -> pd.DataFrame:
    local_csv = Path(__file__).resolve().parent.parent / "data" / "sample_SPY.csv"
    if local_csv.exists():
        log.info(f"Loading local CSV fallback: {local_csv}")
        return _read_local_csv(local_csv)
    log.info("Using synthetic data fallback.")
    return _synthetic_data()


def get_price_data(ticker: str = "SPY", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Fetch Close prices from yfinance; return a long history by default, and fall back if needed.
    Logs 'to today' instead of 'to None' for clarity.
    """
    # Prefer a long history by default
    if start is None:
        start = "2000-01-01"

    end_str = end if end else "today"

    # Try yfinance
    try:
        import yfinance as yf
        log.info(f"Fetching {ticker} via yfinance from {start} to {end_str}...")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            out = df[["Close"]].copy()
            out.index.name = "Date"
            # If too few rows (e.g., very new ticker), fallback to a stable source
            if len(out) >= 200:
                return out
            log.warning(f"yfinance returned only {len(out)} rows (<200). Falling back.")
        else:
            log.warning("yfinance returned empty; falling back.")
    except Exception as e:
        log.warning(f"yfinance failed: {e} — falling back.")

    return _fallback_local_or_synth()
