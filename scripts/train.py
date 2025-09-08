import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

from src.regime_detection import detect_regimes, BULL, BEAR
from src.data_loader import get_price_data
from src.utils import ensure_dir

BEAR_CONF_COLOR = "tab:red"     # confirmed bear
BEAR_CAND_COLOR = "#f4a6a6"     # candidate bear
SMA20_COLOR = "tab:orange"
SMA100_COLOR = "tab:purple"

def _plot_regimes(ticker, df, ax, bear_enter_used, title):
    price = get_price_data(ticker).loc[df.index]
    sma20 = price["Close"].rolling(20, min_periods=5).mean()
    sma100 = price["Close"].rolling(100, min_periods=20).mean()
    price_line, = ax.plot(df.index, df["Close"], lw=1.6, label="Close")
    ax.plot(df.index, sma20, lw=1.2, alpha=0.9, color=SMA20_COLOR, label="SMA 20")
    ax.plot(df.index, sma100, lw=1.2, alpha=0.9, color=SMA100_COLOR, label="SMA 100")

    ymin, ymax = df["Close"].min(), df["Close"].max()

    cand_mask = (df["p_bear_ema"].values >= bear_enter_used)
    ax.fill_between(df.index, ymin, ymax, where=cand_mask, step="pre",
                    alpha=0.12, color=BEAR_CAND_COLOR, label="Bear candidate (light)")

    bear_mask = (df["regime"].values == BEAR)
    ax.fill_between(df.index, ymin, ymax, where=bear_mask, step="pre",
                    alpha=0.25, color=BEAR_CONF_COLOR, label="Bear confirmed (dark)")

    ax.set_ylabel("Price")
    ax.set_title(title)
    legend_handles = [
        Line2D([0],[0], color=price_line.get_color(), lw=2, label="Close"),
        Line2D([0],[0], color=SMA20_COLOR, lw=2, label="SMA 20"),
        Line2D([0],[0], color=SMA100_COLOR, lw=2, label="SMA 100"),
        Patch(facecolor=BEAR_CAND_COLOR, alpha=0.8, label="Bear candidate (light)"),
        Patch(facecolor=BEAR_CONF_COLOR,  alpha=0.8, label="Bear confirmed (dark)"),
        Patch(facecolor="none", edgecolor="none", label="Bull (unshaded)"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)

def _print_last_counts(df, used_enter, used_exit, years):
    if not years or years <= 0: return
    end = df.index.max(); start = end - pd.DateOffset(years=years)
    dfz = df.loc[df.index >= start].copy()
    if dfz.empty:
        print(f"[Counts] Last {years}y window has no rows."); return
    total = len(dfz)
    bear_cand = int((dfz["p_bear_ema"] >= used_enter).sum())
    bull_cand = int((dfz["p_bear_ema"] <= used_exit).sum())
    bear_conf = int((dfz["regime"] == BEAR).sum())
    bull_conf = int((dfz["regime"] == BULL).sum())
    pct = lambda x: round(100*x/total, 2)
    print(f"""
[Last {years}y counts]
  Days: {total}
  Bear (candidate): {bear_cand} ({pct(bear_cand)}%)
  Bear (confirmed): {bear_conf} ({pct(bear_conf)}%)
  Bull (candidate): {bull_cand} ({pct(bull_cand)}%)
  Bull (confirmed): {bull_conf} ({pct(bull_conf)}%)
""")

def _segments_from_mask(dfz: pd.DataFrame, mask: np.ndarray, label: str) -> list[dict]:
    segs = []
    idx = dfz.index; prices = dfz["Close"].values
    n = len(dfz); i = 0
    while i < n:
        if not mask[i]: i += 1; continue
        j = i
        while j + 1 < n and mask[j+1]: j += 1
        p0, p1 = float(prices[i]), float(prices[j])
        segs.append({
            "type": label,
            "start_date": pd.Timestamp(idx[i]).date().isoformat(),
            "start_price": round(p0, 6),
            "end_date": pd.Timestamp(idx[j]).date().isoformat(),
            "end_price": round(p1, 6),
            "n_days": int(j - i + 1),
            "return_pct": round((p1 / p0) - 1.0, 6),
        })
        i = j + 1
    return segs

def _print_and_save_segments(ticker, df, used_enter, used_exit, years, outdir):
    if not years or years <= 0: return
    end = df.index.max(); start = end - pd.DateOffset(years=years)
    dfz = df.loc[df.index >= start].copy()
    if dfz.empty:
        print(f"[Segments] Last {years}y window has no rows."); return
    bear_cand_mask = (dfz["p_bear_ema"].values >= used_enter)
    bull_cand_mask = (dfz["p_bear_ema"].values <= used_exit)
    bear_conf_mask = (dfz["regime"].values == BEAR)
    bull_conf_mask = (dfz["regime"].values == BULL)
    segs = []
    segs += _segments_from_mask(dfz, bear_cand_mask, "bear_candidate")
    segs += _segments_from_mask(dfz, bear_conf_mask, "bear_confirmed")
    segs += _segments_from_mask(dfz, bull_cand_mask, "bull_candidate")
    segs += _segments_from_mask(dfz, bull_conf_mask, "bull_confirmed")
    for s in segs:
        s["direction_ok"] = bool((s["end_price"] < s["start_price"]) if s["type"].startswith("bear")
                                 else (s["end_price"] >= s["start_price"]))
    seg_df = pd.DataFrame(segs, columns=[
        "type","start_date","start_price","end_date","end_price","n_days","return_pct","direction_ok"
    ])
    out_csv = Path(outdir) / f"{ticker}_segments_last{years}y.csv"
    seg_df.to_csv(out_csv, index=False)
    print(f"[Segments] Saved -> {out_csv}  (rows={len(seg_df)})")

    def _pp(group):
        g = seg_df.loc[seg_df["type"] == group]
        print(f"\n[{ticker}] {group} — {len(g)} segments")
        for _, r in g.iterrows():
            print(f"  {r['start_date']} @ {r['start_price']}  →  {r['end_date']} @ {r['end_price']}  "
                  f"days={int(r['n_days'])}  ret={r['return_pct']:+.3%}  direction_ok={bool(r['direction_ok'])}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--n_components", type=int, default=4)

    # smoothing & fixed thresholds
    ap.add_argument("--ema_span", type=int, default=20)
    ap.add_argument("--bear_enter", type=float, default=0.80)
    ap.add_argument("--bear_exit", type=float, default=0.55)

    # auto thresholds
    ap.add_argument("--auto_thresholds", action="store_true")
    ap.add_argument("--enter_q", type=float, default=0.75)
    ap.add_argument("--exit_q", type=float, default=0.55)
    ap.add_argument("--bear_target", type=float, default=None)
    ap.add_argument("--auto_window_years", type=int, default=None)
    ap.add_argument("--min_gap", type=float, default=0.08)
    ap.add_argument("--min_spread", type=float, default=0.05)
    ap.add_argument("--std_floor", type=float, default=5e-3)

    # confirmations & runs
    ap.add_argument("--mom_threshold", type=float, default=0.03)
    ap.add_argument("--ddown_threshold", type=float, default=0.15)
    ap.add_argument("--confirm_days", type=int, default=7)
    ap.add_argument("--bull_mom_threshold", type=float, default=0.01)
    ap.add_argument("--bull_ddown_exit", type=float, default=0.06)
    ap.add_argument("--confirm_days_bull", type=int, default=3)
    ap.add_argument("--min_bear_run", type=int, default=15)
    ap.add_argument("--min_bull_run", type=int, default=5)

    # labeling horizon
    ap.add_argument("--k_forward", type=int, default=10)

    # directional + trend gates
    ap.add_argument("--direction_gate", action="store_true")
    ap.add_argument("--entry_ret_lookback", type=int, default=10)
    ap.add_argument("--entry_ret_thresh", type=float, default=-0.01)  # ≤ -1% over lookback
    ap.add_argument("--entry_dd_thresh", type=float, default=-0.03)   # ≤ -3% from peak
    ap.add_argument("--trend_gate", action="store_true")
    ap.add_argument("--require_close_below_sma100", action="store_true")
    ap.add_argument("--require_sma20_below_sma100", action="store_true")
    ap.add_argument("--trend_exit_cross", action="store_true")
    ap.add_argument("--bear_profit_exit", type=float, default=0.05)

    # strict direction (hard filter)
    ap.add_argument("--strict_direction", action="store_true")
    ap.add_argument("--strict_bear_min_ret", type=float, default=-0.005)
    ap.add_argument("--strict_bear_min_maxdd", type=float, default=-0.03)

    # outputs
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=7.0)
    ap.add_argument("--zoom_years", type=int, default=3)
    args = ap.parse_args()

    df, model = detect_regimes(
        ticker=args.ticker,
        n_components=args.n_components,
        ema_span=args.ema_span,
        bear_enter=args.bear_enter,
        bear_exit=args.bear_exit,
        auto_thresholds=args.auto_thresholds,
        enter_quantile=args.enter_q,
        exit_quantile=args.exit_q,
        bear_target=args.bear_target,
        auto_window_years=args.auto_window_years,
        min_gap=args.min_gap,
        min_spread=args.min_spread,
        std_floor=args.std_floor,
        mom_threshold=args.mom_threshold,
        ddown_threshold=args.ddown_threshold,
        confirm_days=args.confirm_days,
        bull_mom_threshold=args.bull_mom_threshold,
        bull_ddown_exit=args.bull_ddown_exit,
        confirm_days_bull=args.confirm_days_bull,
        min_bear_run=args.min_bear_run,
        min_bull_run=args.min_bull_run,
        k_forward=args.k_forward,
        direction_gate=args.direction_gate,
        entry_ret_lookback=args.entry_ret_lookback,
        entry_ret_thresh=args.entry_ret_thresh,
        entry_dd_thresh=args.entry_dd_thresh,
        trend_gate=args.trend_gate,
        require_close_below_sma100=args.require_close_below_sma100,
        require_sma20_below_sma100=args.require_sma20_below_sma100,
        trend_exit_cross=args.trend_exit_cross,
        bear_profit_exit=args.bear_profit_exit,
        strict_direction=args.strict_direction,
        strict_bear_min_ret=args.strict_bear_min_ret,
        strict_bear_min_maxdd=args.strict_bear_min_maxdd,
    )

    th = getattr(model, "thresholds_", {"bear_enter": args.bear_enter, "bear_exit": args.bear_exit})
    used_enter, used_exit = float(th["bear_enter"]), float(th["bear_exit"])

    ensure_dir(args.outdir)

    # debug CSV
    posterior_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("p_s")]
    cols = ["Close","ret","vol","ret_1","z_ret","mom_20_100","drawdown",
            "state_raw","regime_viterbi","p_bear","p_bear_ema",
            "bear_confirm_trend","bear_confirm_dd","bull_confirm_trend","bull_confirm_dd",
            "bear_confirm","bull_confirm","regime"] + posterior_cols
    dbg = Path(args.outdir) / f"{args.ticker}_debug.csv"
    df[cols].to_csv(dbg); print(f"Saved debug CSV -> {dbg}")
    print(f"Thresholds used: enter={used_enter:.3f}, exit={used_exit:.3f}")

    # title (include knobs)
    title = (f"{args.ticker} — price with regimes "
             f"(k_fwd={args.k_forward}, EMA={args.ema_span}, enter={used_enter:.2f}, exit={used_exit:.2f}, "
             f"min_bear={args.min_bear_run}, min_bull={args.min_bull_run}, "
             f"mom_thr={args.mom_threshold}, dd_thr={args.ddown_threshold}, "
             f"bull_mom_thr={args.bull_mom_threshold}, bull_dd_exit={args.bull_ddown_exit}, "
             f"confirm_bear={args.confirm_days}, confirm_bull={args.confirm_days_bull}, "
             f"dir_gate={args.direction_gate}, lbk={args.entry_ret_lookback}, "
             f"entry_ret_thr={args.entry_ret_thresh}, entry_dd_thr={args.entry_dd_thresh}, "
             f"trend_gate={args.trend_gate}, profit_exit={args.bear_profit_exit}, "
             f"strict={args.strict_direction})")

    # full PNG
    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))
    _plot_regimes(args.ticker, df, ax, bear_enter_used=used_enter, title=title)
    out_full = Path(args.outdir) / f"{args.ticker}_regimes.png"
    fig.savefig(out_full, dpi=args.dpi, bbox_inches="tight"); plt.show()
    print(f"Saved figure -> {out_full}")

    # zoom PNG + counts + segments
    if args.zoom_years and args.zoom_years > 0:
        end = df.index.max(); start = end - pd.DateOffset(years=args.zoom_years)
        dfz = df.loc[df.index >= start]
        if not dfz.empty:
            fig2, ax2 = plt.subplots(figsize=(args.fig_w, args.fig_h))
            _plot_regimes(args.ticker, dfz, ax2, bear_enter_used=used_enter, title=title + f" — last {args.zoom_years}y")
            out_zoom = Path(args.outdir) / f"{args.ticker}_regimes_zoom{args.zoom_years}y.png"
            fig2.savefig(out_zoom, dpi=args.dpi, bbox_inches="tight"); plt.show()
            print(f"Saved zoomed figure -> {out_zoom}")
            _print_last_counts(df, used_enter, used_exit, args.zoom_years)
            _print_and_save_segments(args.ticker, df, used_enter, used_exit, args.zoom_years, args.outdir)

if __name__ == "__main__":
    main()
