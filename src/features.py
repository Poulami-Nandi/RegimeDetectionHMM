import pandas as pd
import numpy as np

def make_features(price_df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Core feature set balanced between volatility and trend:
      - ret: log return
      - vol: rolling std of returns (robust min_periods)
      - ret_1: previous-day return
      - z_ret: z-scored return vs 60d window
      - mom_20_100: normalized SMA(20)-SMA(100) spread
      - drawdown: percent drawdown from rolling peak
    """
    df = price_df.copy()
    df['ret'] = np.log(df['Close']).diff()

    # Vol with minimum periods to avoid dropping everything on short series
    minp = max(5, vol_window // 2)
    df['vol'] = df['ret'].rolling(window=vol_window, min_periods=minp).std().bfill()

    df['ret_1'] = df['ret'].shift(1)

    m60 = df['ret'].rolling(60, min_periods=10)
    df['z_ret'] = (df['ret'] - m60.mean()) / (m60.std() + 1e-8)

    sma20 = df['Close'].rolling(20, min_periods=5).mean()
    sma100 = df['Close'].rolling(100, min_periods=20).mean()
    df['mom_20_100'] = (sma20 - sma100) / (sma100 + 1e-8)

    df['drawdown'] = (df['Close'] / df['Close'].cummax()) - 1.0

    df = df.dropna()
    return df[['ret', 'vol', 'ret_1', 'z_ret', 'mom_20_100', 'drawdown']]
