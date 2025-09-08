import argparse
import pandas as pd
import numpy as np
from src.regime_detection import detect_regimes, BULL, BEAR

def backtest_long_in_bull(df: pd.DataFrame) -> pd.DataFrame:
    strat_ret = df['ret'] * (df['regime'] == BULL).astype(int)
    eq = (1 + strat_ret).cumprod()
    buyhold = (1 + df['ret']).cumprod()
    out = pd.DataFrame({'eq_strategy': eq, 'eq_buyhold': buyhold}, index=df.index)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', default='SPY')
    ap.add_argument('--n_components', type=int, default=2)
    args = ap.parse_args()

    df, _ = detect_regimes(ticker=args.ticker, n_components=args.n_components)
    res = backtest_long_in_bull(df)
    print(res.tail())
    print('\nStrategy final: %.3f | Buy&Hold final: %.3f' % (res['eq_strategy'].iloc[-1], res['eq_buyhold'].iloc[-1]))

if __name__ == '__main__':
    main()
