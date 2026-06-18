import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime
from pathlib import Path

from core.metrics import (
    daily_return,
    daily_std,
    daily_sharpe,
    annual_max_drawdown,
    annual_pnl,
    performance_summary,
    compute_metrics
)
from data.loader import fetch_pair_prices
from strategies.pairs_trading import OUpairTrader
from backtest.engine import BacktestEngine


def print_metrics(metrics):
    """Pretty-print metrics to console."""
    print("\n" + "=" * 60)
    print("           BACKTEST PERFORMANCE REPORT")
    print("=" * 60)
    print(f"{'Total Trades:':<30} {metrics['total_trades']}")
    print(f"{'Winning Trades:':<30} {metrics['winning_trades']}")
    print(f"{'Losing Trades:':<30} {metrics['losing_trades']}")
    print(f"{'Win Rate:':<30} {metrics['win_rate']:.2f}%")
    print("-" * 60)
    print(f"{'Avg Trade PnL:':<30} ${metrics['avg_trade_pnl']:,.2f}")
    print(f"{'Avg Win:':<30} ${metrics['avg_win']:,.2f}")
    print(f"{'Avg Loss:':<30} ${metrics['avg_loss']:,.2f}")
    print(f"{'Profit Factor:':<30} {metrics['profit_factor']:.2f}")
    print(f"{'Max Trade PnL:':<30} ${metrics['max_trade_pnl']:,.2f}")
    print(f"{'Min Trade PnL:':<30} ${metrics['min_trade_pnl']:,.2f}")
    print("-" * 60)
    print(f"{'Total Net PnL:':<30} ${metrics['total_pnl']:,.2f}")
    print(f"{'Total Return:':<30} {metrics['total_return_pct']:.2f}%")
    print(f"{'Avg Daily Return:':<30} {metrics['avg_daily_return']:.4f}%")
    print(f"{'Avg Daily Std:':<30} {metrics['avg_daily_std']:.4f}%")
    print(f"{'Sharpe Ratio:':<30} {metrics['sharpe_ratio']:.3f}")
    print(f"{'Max Drawdown:':<30} ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    print(f"{'Calmar Ratio:':<30} {metrics['calmar_ratio']:.3f}")
    print("-" * 60)
    print("Annual PnL:")
    for year, pnl in metrics['annual_pnl'].items():
        print(f"  {year}: ${pnl:,.2f}")
    print("Annual Max Drawdown:")
    for year, dd in metrics['annual_max_dd'].items():
        print(f"  {year}: ${dd:,.2f}")
    print("=" * 60)

def generate_report(results_df, trades_df, metrics, strategy, pair_id, initial_capital,
                    formation_data, trading_data, output_dir="reports"):
    """
    Generate a single comprehensive figure containing all key backtest visuals.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"backtest_report_{pair_id[0]}_{pair_id[1]}_{timestamp}"

    # --- Prepare data ---
    results_df = results_df.copy()
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.set_index('date')

    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

    # --- Create comprehensive figure ---
    fig = plt.figure(figsize=(22, 30))
    gs = GridSpec(5, 3, figure=fig, height_ratios=[2.5, 1.2, 1.5, 1.5, 1.5],
                  hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Backtest Report — {pair_id[0]}/{pair_id[1]}\n"
        f"Total Return: {metrics['total_return_pct']:.2f}% | "
        f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
        f"Max DD: {metrics['max_drawdown_pct']:.2f}% | "
        f"Win Rate: {metrics['win_rate']:.1f}%"
        f"\nDaterange : {results_df.index[0].date()} - {results_df.index[-1].date()}",
        fontsize=16, fontweight='bold', y=0.995
    )

    # 1. Equity Curve (top, full width)
    ax_equity = fig.add_subplot(gs[0, :])
    ax_equity.plot(results_df.index, results_df['capital'],
                   color='#2E86AB', linewidth=1.5, label='Capital')
    ax_equity.axhline(y=initial_capital, color='gray', linestyle='--',
                      alpha=0.7, label='Initial Capital')
    ax_equity.fill_between(results_df.index, results_df['capital'], initial_capital,
                           where=(results_df['capital'] > initial_capital),
                           color='green', alpha=0.15)
    ax_equity.fill_between(results_df.index, results_df['capital'], initial_capital,
                           where=(results_df['capital'] < initial_capital),
                           color='red', alpha=0.15)
    ax_equity.set_ylabel("Capital ($)")
    ax_equity.legend(loc='upper left')
    ax_equity.grid(True, alpha=0.3)
    ax_equity.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_equity.set_title("Equity Curve", fontsize=12, fontweight='bold', loc='left')

    # 2. Drawdown (second row, full width, shared x-axis)
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_equity)
    if not trades_df.empty:
        daily = trades_df.groupby('exit_date')['net_pnl'].sum()
        equity = daily.cumsum() + initial_capital
        equity.index = pd.to_datetime(equity.index)
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max * 100
    else:
        dd = pd.Series(0, index=results_df.index)

    ax_dd.fill_between(dd.index, dd, 0, color='crimson', alpha=0.4)
    ax_dd.plot(dd.index, dd, color='crimson', linewidth=1)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_dd.set_title("Drawdown", fontsize=12, fontweight='bold', loc='left')

    # 3. Daily PnL
    ax_daily = fig.add_subplot(gs[2, 0])
    if not trades_df.empty:
        daily_pnl = trades_df.groupby('exit_date')['net_pnl'].sum()
        colors = ['green' if x > 0 else 'crimson' for x in daily_pnl]
        ax_daily.bar(daily_pnl.index, daily_pnl, color=colors, alpha=0.7, width=1.0)
        ax_daily.axhline(y=0, color='black', linewidth=0.8)
    ax_daily.set_title("Daily PnL", fontsize=11, fontweight='bold')
    ax_daily.set_ylabel("PnL ($)")
    ax_daily.grid(True, alpha=0.3, axis='y')
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_daily.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_daily.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Trade PnL Histogram
    ax_hist = fig.add_subplot(gs[2, 1])
    if not trades_df.empty:
        bins = max(20, int(np.sqrt(len(trades_df))))
        ax_hist.hist(trades_df['net_pnl'], bins=bins, color='steelblue',
                     edgecolor='white', alpha=0.8)
        ax_hist.axvline(x=metrics['avg_trade_pnl'], color='orange',
                        linestyle='--', linewidth=2,
                        label=f"Avg: ${metrics['avg_trade_pnl']:,.2f}")
        ax_hist.axvline(x=0, color='black', linewidth=1)
    ax_hist.set_title("Trade PnL Distribution", fontsize=11, fontweight='bold')
    ax_hist.set_xlabel("Trade PnL ($)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3, axis='y')

    # 5. Normalized Performance
    ax_norm = fig.add_subplot(gs[2, 2])
    s1 = trading_data.iloc[:, 0]
    s2 = trading_data.iloc[:, 1]
    s1_norm = s1 / s1.iloc[0] * 100
    s2_norm = s2 / s2.iloc[0] * 100
    strat_norm = results_df['capital'] / initial_capital * 100
    ax_norm.plot(s1_norm.index, s1_norm, label=pair_id[0],
                 color='#E63946', linewidth=1.2)
    ax_norm.plot(s2_norm.index, s2_norm, label=pair_id[1],
                 color='#457B9D', linewidth=1.2)
    ax_norm.plot(strat_norm.index, strat_norm, label='Strategy',
                 color='#2A9D8F', linewidth=1.5)
    ax_norm.set_ylabel("Normalized (Base=100)")
    ax_norm.legend(loc='upper left', fontsize=8)
    ax_norm.grid(True, alpha=0.3)
    ax_norm.set_title("Normalized Performance", fontsize=11, fontweight='bold')

    # 6. Spread
    ax_spread = fig.add_subplot(gs[3, 0])
    spread = s1 - strategy.b_ratio * s2
    ax_spread.plot(spread.index, spread, color='#6A4C93', linewidth=1)
    ax_spread.axhline(y=strategy.mu, color='orange', linestyle='--',
                      label=f"mu = {strategy.mu:.2f}")
    ax_spread.axhline(y=strategy.theta, color='red', linestyle='--',label=f'mu = {strategy.theta:.2f}')
    ax_spread.fill_between(spread.index, spread, strategy.mu,
                           where=(spread > strategy.mu), color='green', alpha=0.1)
    ax_spread.fill_between(spread.index, spread, strategy.mu,
                           where=(spread < strategy.mu), color='red', alpha=0.1)
    ax_spread.set_ylabel("Spread ($)")
    ax_spread.legend(fontsize=8)
    ax_spread.grid(True, alpha=0.3)
    ax_spread.set_title(
        f"Spread: {pair_id[0]} - {strategy.b_ratio:.3f}×{pair_id[1]}",
        fontsize=11, fontweight='bold')

    # 7. Signal Overlay on Spread
    ax_signals = fig.add_subplot(gs[3, 1:], sharex=ax_spread)
    ax_signals.plot(spread.index, spread, color='#6A4C93',
                    linewidth=1, alpha=0.6)
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            color = 'green' if row['side'] == 'long_spread' else 'red'
            ax_signals.axvspan(row['entry_date'], row['exit_date'],
                               color=color, alpha=0.15)
    ax_signals.set_ylabel("Spread ($)")
    ax_signals.set_xlabel("Date")
    ax_signals.set_title("Trade Positions (Green=Long, Red=Short)",
                         fontsize=11, fontweight='bold')
    ax_signals.grid(True, alpha=0.3)
    ax_signals.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_signals.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_signals.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 8. Rolling Sharpe
    ax_sharpe = fig.add_subplot(gs[4, 0])
    if not trades_df.empty:
        roll_sharpe = daily_sharpe(trades_df, rf=0.0, ann=252)
        ax_sharpe.plot(roll_sharpe.index, roll_sharpe,
                       color='#264653', linewidth=1.2)
        ax_sharpe.axhline(y=0, color='black', linewidth=0.8)
        ax_sharpe.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax_sharpe.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax_sharpe.set_ylabel("Sharpe Ratio")
        ax_sharpe.set_title("Expanding Sharpe Ratio",
                            fontsize=11, fontweight='bold')
        ax_sharpe.grid(True, alpha=0.3)
    else:
        ax_sharpe.text(0.5, 0.5, "No trades", ha='center', va='center',
                       transform=ax_sharpe.transAxes)
        ax_sharpe.set_title("Expanding Sharpe Ratio",
                            fontsize=11, fontweight='bold')

    # 9. Rolling Volatility
    ax_vol = fig.add_subplot(gs[4, 1])
    if not trades_df.empty:
        roll_std = daily_std(trades_df)
        ax_vol.plot(roll_std.index, roll_std * 100,
                    color='#E76F51', linewidth=1.2)
        ax_vol.set_ylabel("Volatility (%)")
        ax_vol.set_xlabel("Date")
        ax_vol.set_title("Expanding Daily Volatility",
                         fontsize=11, fontweight='bold')
        ax_vol.grid(True, alpha=0.3)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_vol.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax_vol.text(0.5, 0.5, "No trades", ha='center', va='center',
                    transform=ax_vol.transAxes)
        ax_vol.set_title("Expanding Daily Volatility",
                         fontsize=11, fontweight='bold')

    # 10. Monthly Returns Heatmap
    ax_heatmap = fig.add_subplot(gs[4, 2])
    if not trades_df.empty:
        daily = trades_df.groupby('exit_date')['net_pnl'].sum()
        daily.index = pd.to_datetime(daily.index)
        equity = daily.cumsum() + initial_capital
        monthly_returns = equity.resample('ME').last().pct_change().dropna() * 100
        monthly_df = monthly_returns.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        im = ax_heatmap.imshow(pivot.values, cmap='RdYlGn',
                               aspect='auto', vmin=-10, vmax=10)
        ax_heatmap.set_xticks(range(len(pivot.columns)))
        ax_heatmap.set_xticklabels(
            [pd.to_datetime(f"2020-{m:02d}-01").strftime('%b')
             for m in pivot.columns], fontsize=8)
        ax_heatmap.set_yticks(range(len(pivot.index)))
        ax_heatmap.set_yticklabels(pivot.index, fontsize=8)
        ax_heatmap.set_title("Monthly Returns (%)",
                             fontsize=11, fontweight='bold')

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax_heatmap.text(j, i, f"{val:.1f}", ha="center",
                                    va="center", color="black", fontsize=7)
    else:
        ax_heatmap.text(0.5, 0.5, "No trades", ha='center', va='center',
                        transform=ax_heatmap.transAxes)
        ax_heatmap.set_title("Monthly Returns (%)",
                             fontsize=11, fontweight='bold')

    #plt.tight_layout(rect=[0, 0, 1, 0.99])
    report_path = os.path.join(output_dir, f"{report_name}_comprehensive.png")
    validation = input("Do you want to save the report ? (Y/N)").upper()
    if validation == 'Y' :
        fig.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Comprehensive report saved to: {report_path}")
    else : 
        plt.show()

    return report_path


def main():
    config = {
        "trading": {
            "M": 63,
            "K": 1.5,
            "stop_loss_r": 0.05,
            "slippage": 0.0005,
            "commission": 0.001,
        },
        "ou": {
            "hedge_ratio_grid": [0.1, 2.0, 0.01]
        }
    }

    tickers = ["MS", "GS"]
    formation_start = "2021-01-01"
    formation_end = "2021-12-31"
    trading_end = "2022-12-31"
    interval = "1d"
    initial_capital = 100_000.0

    print(f"Fetching data for pair: {tickers}...")
    full_df = fetch_pair_prices(tickers, start=formation_start,
                                end=trading_end, itv=interval)

    formation_data = full_df.loc[formation_start:formation_end].copy()
    trading_data = full_df.loc[formation_end:trading_end].copy()

    if formation_data.empty or trading_data.empty:
        print("Error: Empty dataframes. Check dates or connection.")
        return

    print(f"Formation: {len(formation_data)} rows | "
          f"Trading: {len(trading_data)} rows")

    print("\nFitting OU parameters...")
    strategy = OUpairTrader(config)
    strategy.fit(formation_data)

    print(f"B={strategy.b_ratio:.4f}, mu={strategy.mu:.4f}, "
          f"theta={strategy.theta:.4f}, sigma={strategy.sigma:.4f}")

    print("\nRunning backtest...")
    engine = BacktestEngine(initial_capital=initial_capital, config=config)
    results_df = engine.run(pair_id=tickers, strategy=strategy,
                            price_data=trading_data)

    trades_df = pd.DataFrame(engine.trade_log) if engine.trade_log else pd.DataFrame()

    # Compute metrics
    metrics = compute_metrics(trades_df, results_df, initial_capital)
    print_metrics(metrics)

    # Generate report
    generate_report(
        results_df=results_df,
        trades_df=trades_df,
        metrics=metrics,
        strategy=strategy,
        pair_id=tickers,
        initial_capital=initial_capital,
        formation_data=formation_data,
        trading_data=trading_data,
        output_dir="reports"
    )



if __name__ == "__main__":
    main()
