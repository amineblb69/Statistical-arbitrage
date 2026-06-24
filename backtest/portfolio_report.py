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

from core.metrics import (
    compute_metrics,
    compute_metrics_from_equity,
    equity_returns,
    equity_drawdown,
)
from strategies.multipair_framework import MultiPairFramework
from data.loader import fetch_pair_prices

class PortfolioReport:

    def __init__(self, config, pairs, initial_capital : float = 100_000.0):

        self.config = config   
        self.pairs  = [tuple(p) for p in pairs]
        self.initial_capital = initial_capital
        
        data_cfg = config.get("data",{})
        self.formation_start = data_cfg.get("formation_start","2022-12-31")
        self.formation_end   = data_cfg.get("formation_end","2023-12-31")
        self.trading_start   = data_cfg.get("trading_start","2023-12-31")
        self.trading_end     = data_cfg.get("trading_end","2025-12-31")
        self.interval        = data_cfg.get("interval","1d")

        trading_cfg = config.get("trading", {})
        ou_cfg      = config.get("ou", {})
        div_cfg     = config.get("diversification",{})

        self.framework_config = {
            "ou":{
                "hedge_ratio_grid": [0.1, 2.0, 0.01],
                "lookback_L": 252,
            },
            "method": div_cfg.get("method_choosed","EW"),
            "L" : ou_cfg.get("loockback_L", 252),
            "Lr" : div_cfg.get("rebalance_windox", 63),
            "M" : trading_cfg.get("M", 63),
            "K" : trading_cfg.get("K", 1.5),
            "r" : trading_cfg.get("stop_loss_r", 0.1),
            "trading" : { 
                "M" : trading_cfg.get("M", 63),
                "K" : trading_cfg.get("K", 1.5),
                "stop_loss_r" : trading_cfg.get("stop_loss_r", 0.1),
                "slippage" : trading_cfg.get("slippage", 0.0005),
                "commission": trading_cfg.get("commission", 0.1)
                
            }
        }

        self.framework          = None 
        self.price_data         = None
        self.portfolio_log      = None
        self.pair_daily_logs    = None
        self.equity_curve       = None
        self.portfolio_metrics  = None
        self.pair_metrics       = None
        self.pair_contributions = None

    def _load_price_data(self) -> dict:
        price_data = {}
        for pair in self.pairs:
            full_df = fetch_pair_prices(
                pair,
                start=self.formation_start,
                end=self.trading_end,
                itv=self.interval,
            )
            full_df.columns = [f"{pair[0]}_close", f"{pair[1]}_close"]
            price_data[pair] = full_df

        return price_data
    
    def run(self):

        self.price_data = self._load_price_data()
        self.framework  = MultiPairFramework(
            config        = self.framework_config,
            pairs         = self.pairs,
            total_capital = self.initial_capital
        )

        self.portfolio_log = self.framework.run_backtest(
            price_data    = self.price_data,
            trading_start = self.trading_start,
            trading_end   = self.trading_end
        )

        if self.portfolio_log.empty :
            raise ValueError("Portfolio backtest produced no rows. Check the params fam.")
        
        self.pair_daily_logs = {
            pair: pd.DataFrame(self.framework.pair_daily_logs.get(pair, []))
            for pair in self.pairs
            }
        
        for pair in self.pairs :
            plog = self.portfolio_log.copy()
            plog["date"] = pd.to_datetime(plog["date"])
            self.equity_curve = plog.set_index("date")["capital"].sort_index()

            self.portfolio_metrics = compute_metrics_from_equity(self.equity_curve,self.initial_capital)

            self._compute_pair_breakdown()

            return self
        
    def _compute_pair_breakdown(self):
             
            """
            For each pair, compute two distinct things (see conceptual note):
    
            1. Standalone performance: as if this pair were the only thing
            traded. Uses that pair's own trade_log through the *existing*,
            untouched compute_metrics() (trade-log path) — same function
            pair_report.py already uses.
            2. Dollar contribution to the portfolio: sum of realized PnL this
            pair generated inside the multi-pair run, accounting for
            whatever capital it was actually allocated at each rebalance.
            This comes from pair_daily_logs, not from compute_metrics().
            """

            self.pair_metrics       = {}
            self.pair_contributions = {}

            for pair in self.pairs :

                engine = self.framework.pair_engines.get(pair)
                trades_df = (
                    pd.DataFrame(engine.trade_log)
                    if engine is not None and engine.trade_log
                    else pd.DataFrame()
                )

                ref_capital = (
                    engine.initial_capital if engine is not None
                    else self.initial_capital / len(self.pairs)
                )

                self.pair_metrics[pair] = compute_metrics(trades_df, results_df=None,initial_capital=ref_capital)
                
                daily_log = self.pair_daily_logs.get(pair, pd.DataFrame())
                self.pair_contributions[pair] = (
                float(daily_log["pnl"].sum()) if not daily_log.empty else 0.0 )

    def print_metrics(self):
        if self.portfolio_metrics is None:
            raise RuntimeError("Call .run() before .print_metrics().")
 
        m = self.portfolio_metrics
        print("\n" + "=" * 64)
        print("           PORTFOLIO BACKTEST PERFORMANCE REPORT")
        print("=" * 64)
        print(f"{'Pairs:':<30} {', '.join(f'{a}/{b}' for a, b in self.pairs)}")
        print(f"{'Allocation Method:':<30} {self.framework_config['method']}")
        print(f"{'Initial Capital:':<30} ${self.initial_capital:,.2f}")
        print("-" * 64)
        print(f"{'Total Net PnL:':<30} ${m['total_pnl']:,.2f}")
        print(f"{'Total Return:':<30} {m['total_return_pct']:.2f}%")
        print(f"{'Avg Daily Return:':<30} {m['avg_daily_return']:.4f}%")
        print(f"{'Avg Daily Std:':<30} {m['avg_daily_std']:.4f}%")
        print(f"{'Sharpe Ratio:':<30} {m['sharpe_ratio']:.3f}")
        print(f"{'Max Drawdown:':<30} ${m['max_drawdown']:,.2f} ({m['max_drawdown_pct']:.2f}%)")
        print(f"{'Calmar Ratio:':<30} {m['calmar_ratio']:.3f}")
        print("-" * 64)
        print("Annual PnL (portfolio):")
        for year, pnl in m["annual_pnl"].items():
            print(f"  {year}: ${pnl:,.2f}")
        print("-" * 64)
        print("Per-Pair Breakdown:")
        print(f"  {'Pair':<14}{'Contribution $':>16}{'Sharpe':>10}{'Win Rate':>11}{'Trades':>9}")
        for pair in self.pairs:
            pm = self.pair_metrics[pair]
            contrib = self.pair_contributions[pair]
            label = f"{pair[0]}/{pair[1]}"
            print(
                f"  {label:<14}{contrib:>16,.2f}{pm['sharpe_ratio']:>10.3f}"
                f"{pm['win_rate']:>10.1f}%{pm['total_trades']:>9}"
            )
        print("=" * 64)
 
    # ------------------------------------------------------------------
    # Visual report
    # ------------------------------------------------------------------
    def generate_report(self, output_dir="reports", show=True):
        """
        Build a single comprehensive figure with portfolio-level and
        per-pair panels. Mirrors the structure of pair_report.generate_report().
        """
        if self.equity_curve is None:
            raise RuntimeError("Call .run() before .generate_report().")
 
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_tag = "_".join(f"{a}-{b}" for a, b in self.pairs)
        report_name = f"portfolio_report_{pair_tag}_{timestamp}"
 
        m = self.portfolio_metrics
        n_pairs = len(self.pairs)
 
        fig = plt.figure(figsize=(22, 24))
        gs = GridSpec(4, 3, figure=fig, height_ratios=[2.2, 1.2, 1.4, 1.4],
                      hspace=0.35, wspace=0.3)
        fig.suptitle(
            f"Portfolio Report — {', '.join(f'{a}/{b}' for a, b in self.pairs)}\n"
            f"Total Return: {m['total_return_pct']:.2f}% | "
            f"Sharpe: {m['sharpe_ratio']:.3f} | "
            f"Max DD: {m['max_drawdown_pct']:.2f}% | "
            f"Method: {self.framework_config['method']}",
            fontsize=15, fontweight='bold', y=0.995
        )
 
        # 1. Portfolio equity curve
        ax_eq = fig.add_subplot(gs[0, :])
        ax_eq.plot(self.equity_curve.index, self.equity_curve.values,
                   color='#2E86AB', linewidth=1.5, label='Portfolio Capital')
        ax_eq.axhline(y=self.initial_capital, color='gray', linestyle='--',
                      alpha=0.7, label='Initial Capital')
        ax_eq.fill_between(self.equity_curve.index, self.equity_curve.values,
                           self.initial_capital,
                           where=(self.equity_curve.values > self.initial_capital),
                           color='green', alpha=0.15)
        ax_eq.fill_between(self.equity_curve.index, self.equity_curve.values,
                           self.initial_capital,
                           where=(self.equity_curve.values < self.initial_capital),
                           color='red', alpha=0.15)
        ax_eq.set_ylabel("Capital ($)")
        ax_eq.legend(loc='upper left')
        ax_eq.grid(True, alpha=0.3)
        ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax_eq.set_title("Portfolio Equity Curve", fontsize=12, fontweight='bold', loc='left')
 
        # 2. Portfolio drawdown
        ax_dd = fig.add_subplot(gs[1, :], sharex=ax_eq)
        dd_pct = equity_drawdown(self.equity_curve) / self.equity_curve.cummax() * 100
        ax_dd.fill_between(dd_pct.index, dd_pct.values, 0, color='crimson', alpha=0.4)
        ax_dd.plot(dd_pct.index, dd_pct.values, color='crimson', linewidth=1)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.grid(True, alpha=0.3)
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax_dd.set_title("Portfolio Drawdown", fontsize=12, fontweight='bold', loc='left')
 
        # 3. Per-pair dollar contribution
        ax_contrib = fig.add_subplot(gs[2, 0])
        labels = [f"{a}/{b}" for a, b in self.pairs]
        values = [self.pair_contributions[p] for p in self.pairs]
        colors = ['green' if v > 0 else 'crimson' for v in values]
        ax_contrib.bar(labels, values, color=colors, alpha=0.75)
        ax_contrib.axhline(y=0, color='black', linewidth=0.8)
        ax_contrib.set_title("PnL Contribution by Pair", fontsize=11, fontweight='bold')
        ax_contrib.set_ylabel("PnL ($)")
        ax_contrib.grid(True, alpha=0.3, axis='y')
        plt.setp(ax_contrib.xaxis.get_majorticklabels(), rotation=30, ha='right')
 
        # 4. Per-pair standalone Sharpe
        ax_sharpe_bar = fig.add_subplot(gs[2, 1])
        sharpes = [self.pair_metrics[p]['sharpe_ratio'] for p in self.pairs]
        colors_s = ['green' if s > 0 else 'crimson' for s in sharpes]
        ax_sharpe_bar.bar(labels, sharpes, color=colors_s, alpha=0.75)
        ax_sharpe_bar.axhline(y=0, color='black', linewidth=0.8)
        ax_sharpe_bar.set_title("Standalone Sharpe by Pair", fontsize=11, fontweight='bold')
        ax_sharpe_bar.set_ylabel("Sharpe Ratio")
        ax_sharpe_bar.grid(True, alpha=0.3, axis='y')
        plt.setp(ax_sharpe_bar.xaxis.get_majorticklabels(), rotation=30, ha='right')
 
        # 5. Allocation weights over rebalancing stages
        ax_weights = fig.add_subplot(gs[2, 2])
        weights_table = self.framework.get_weights_table()
        if not weights_table.empty:
            weights_table.plot(kind='bar', stacked=True, ax=ax_weights,
                               colormap='tab10', alpha=0.85, legend=False)
            ax_weights.set_title("Capital Allocation by Stage", fontsize=11, fontweight='bold')
            ax_weights.set_xlabel("Rebalance Stage")
            ax_weights.set_ylabel("Capital ($)")
            ax_weights.legend(fontsize=6, loc='upper right')
        else:
            ax_weights.text(0.5, 0.5, "No rebalance stages", ha='center', va='center',
                            transform=ax_weights.transAxes)
            ax_weights.set_title("Capital Allocation by Stage", fontsize=11, fontweight='bold')
 
        # 6. Per-pair equity curves overlay (normalized to 100)
        ax_pair_eq = fig.add_subplot(gs[3, :2])
        for pair in self.pairs:
            log = self.pair_daily_logs.get(pair, pd.DataFrame())
            if log.empty:
                continue
            log = log.copy()
            log['date'] = pd.to_datetime(log['date'])
            series = log.set_index('date')['capital'].sort_index()
            norm = series / series.iloc[0] * 100
            ax_pair_eq.plot(norm.index, norm.values, linewidth=1.2,
                            label=f"{pair[0]}/{pair[1]}")
        ax_pair_eq.set_ylabel("Normalized Capital (Base=100)")
        ax_pair_eq.set_title("Per-Pair Capital Curves (Normalized)",
                             fontsize=11, fontweight='bold')
        ax_pair_eq.legend(fontsize=8, loc='upper left')
        ax_pair_eq.grid(True, alpha=0.3)
        ax_pair_eq.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_pair_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_pair_eq.xaxis.get_majorticklabels(), rotation=45, ha='right')
 
        # 7. Correlation heatmap between pairs' daily returns
        ax_corr = fig.add_subplot(gs[3, 2])
        ret_frame = {}
        for pair in self.pairs:
            log = self.pair_daily_logs.get(pair, pd.DataFrame())
            if log.empty:
                continue
            log = log.copy()
            log['date'] = pd.to_datetime(log['date'])
            series = log.set_index('date')['capital'].sort_index()
            ret_frame[f"{pair[0]}/{pair[1]}"] = equity_returns(series)
        if len(ret_frame) >= 2:
            ret_df = pd.DataFrame(ret_frame).dropna(how='all')
            corr = ret_df.corr()
            im = ax_corr.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
            ax_corr.set_xticks(range(len(corr.columns)))
            ax_corr.set_xticklabels(corr.columns, fontsize=7, rotation=45, ha='right')
            ax_corr.set_yticks(range(len(corr.index)))
            ax_corr.set_yticklabels(corr.index, fontsize=7)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    ax_corr.text(j, i, f"{corr.values[i, j]:.2f}", ha='center',
                                va='center', color='black', fontsize=7)
            ax_corr.set_title("Pair Return Correlation", fontsize=11, fontweight='bold')
        else:
            ax_corr.text(0.5, 0.5, "Need >=2 pairs\nwith data", ha='center', va='center',
                        transform=ax_corr.transAxes)
            ax_corr.set_title("Pair Return Correlation", fontsize=11, fontweight='bold')

        report_path = os.path.join(output_dir, f"{report_name}.png")
        
        if show:
            plt.show()
        else:
            plt.close(fig)

        save = input('Do you want the report to be saved ? (Y/N) : ').upper()
        if save == 'Y':
            fig.savefig(report_path, dpi=150, bbox_inches='tight')
            print(f"Portfolio report saved to: {report_path}")
        
        return report_path
 
def main():

        config = {
            "data": {
                "formation_start": "2022-01-01",
                "formation_end": "2022-12-31",
                "trading_start": "2023-01-01",
                "trading_end": "2024-12-31",
                "interval": "1d",
            },
            "ou": {
                "hedge_ratio_grid": [0.1, 2.0, 0.01],
                "lookback_L": 252,
            },
            "trading": {
                "M": 30,
                "K": 1.5,
                "stop_loss_r": 0.1,
                "slippage": 0.0005,
                "commission": 0.001,
            },
            "diversification": {
                "rebalance_window": 63,
                "method_choosed": "MRB",  
            },
        }

        pairs = [["MS","GS"],["V","MA"],["UAL","DAL"],["NVDA","AMD"],["WM","RSG"],["CVX","XOM"]]
        initial_capital = 100_000.0

        report = PortfolioReport(config, pairs, initial_capital)
        report.run()
        report.print_metrics()
        report.generate_report(output_dir="reports")

if __name__ == "__main__" :
    main()


        
    


