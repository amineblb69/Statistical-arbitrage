"""
backtest/comparison_report.py

Reproduces the paper's Figure 2.4 (2x2 grid of cumulative-return curves for
EW/MRB/MRR/MVA at k in {0.5, 0.75, 1.0, 1.25}) and Table 2.5 (per-K block of
DailyRet / DailyStd / DailySR / AnnMDD / AnnPnL by method).

Reuses:
    - MultiPairFramework           (strategies.multipair_framework)
    - equity-series metrics        (core.metrics)
    - fetch_pair_prices            (data.loader)

Design:
    Fetch price data ONCE, then run 4 methods x 4 K values = 16 backtests
    on the same data with a deep-copied config each time. Everything else
    (formation/trading dates, initial capital, L, Lr, M, r, transaction
    costs) is held constant to isolate the effect of method x K.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from core.metrics import (
    equity_returns,
    equity_daily_std,
    equity_daily_sharpe,
    equity_annual_max_drawdown,
    equity_annual_pnl,
)
from strategies.multipair_framework import MultiPairFramework
from data.loader import fetch_pair_prices


# Paper's color mapping for Figure 2.4
METHOD_COLORS = {
    "EW":  "#1f77b4",   # blue
    "MRB": "#ff7f0e",   # orange
    "MRR": "#2ca02c",   # green
    "MVA": "#d62728",   # red
}
METHOD_LABELS = {"EW": "Equal weight", "MRB": "MRB", "MRR": "MRR", "MVA": "MVA"}


class ComparisonReport:
    """
    Orchestrates a method x K grid of backtests and produces the paper's
    Figure 2.4 and Table 2.5 equivalents.
    """

    def __init__(
        self,
        config: dict,
        pairs: list,
        initial_capital: float = 100_000.0,
        methods: list = ("EW", "MRB", "MRR", "MVA"),
        k_values: list = (0.5, 0.75, 1.0, 1.25),
    ):
        self.base_config = config
        self.pairs = [tuple(p) for p in pairs]
        self.initial_capital = float(initial_capital)
        self.methods = list(methods)
        self.k_values = list(k_values)

        data_cfg = config.get("data", {})
        self.formation_start = data_cfg.get("formation_start")
        self.formation_end = data_cfg.get("formation_end")
        self.trading_start = data_cfg.get("trading_start")
        self.trading_end = data_cfg.get("trading_end")
        self.interval = data_cfg.get("interval", "1d")

        # Filled by run()
        self.price_data = None                         # {pair: DataFrame}
        self.equity_curves = {}                        # {(K, method): pd.Series}
        self.metrics_table = None                      # long-format DataFrame

    # ------------------------------------------------------------------
    # 1. Data (fetched once, reused across all 16 runs)
    # ------------------------------------------------------------------
    def _load_price_data(self) -> dict:
        price_data = {}
        for pair in self.pairs:
            full_df = fetch_pair_prices(
                list(pair),
                start=self.formation_start,
                end=self.trading_end,
                itv=self.interval,
            )
            full_df.columns = [f"{pair[0]}_close", f"{pair[1]}_close"]
            price_data[pair] = full_df
        return price_data

    # ------------------------------------------------------------------
    # 2. Framework-config builder for a given (method, K) run
    # ------------------------------------------------------------------
    def _build_framework_config(self, method: str, k: float) -> dict:
        trading_cfg = self.base_config.get("trading", {})
        ou_cfg = self.base_config.get("ou", {})
        div_cfg = self.base_config.get("diversification", {})

        # Deep-copy the base config so state cannot leak between runs
        return copy.deepcopy({
            "ou": {
                "hedge_ratio_grid": ou_cfg.get("hedge_ratio_grid", [-2.0, 2.0, 0.01]),
                "lookback_L":       ou_cfg.get("lookback_L", 252),
            },
            "method": method,
            "L":  ou_cfg.get("lookback_L", 252),
            "Lr": div_cfg.get("rebalance_window", 63),
            "M":  trading_cfg.get("M", 63),
            "K":  float(k),
            "r":  trading_cfg.get("stop_loss_r", 0.1),
            "trading": {
                "M":           trading_cfg.get("M", 63),
                "K":           float(k),
                "stop_loss_r": trading_cfg.get("stop_loss_r", 0.1),
                "slippage":    trading_cfg.get("slippage", 0.0),
                "commission":  trading_cfg.get("commission", 0.0),
            },
        })

    # ------------------------------------------------------------------
    # 3. Main entry point — runs all 16 backtests
    # ------------------------------------------------------------------
    def run(self):
        print(f"[ComparisonReport] Fetching price data for {len(self.pairs)} pairs...")
        self.price_data = self._load_price_data()

        rows = []
        for k in self.k_values:
            for method in self.methods:
                print(f"[ComparisonReport] Running method={method:<3s} K={k}")
                fw_cfg = self._build_framework_config(method, k)

                framework = MultiPairFramework(
                    config=fw_cfg,
                    pairs=self.pairs,
                    total_capital=self.initial_capital,
                )

                # Same price data passed by reference every run (AC3)
                framework.run_backtest(
                    price_data=self.price_data,
                    trading_start=self.trading_start,
                    trading_end=self.trading_end,
                )

                portfolio_log = pd.DataFrame(framework.portfolio_log)
                if portfolio_log.empty:
                    print(f"  !! empty portfolio_log for {method}/{k}, skipping")
                    continue

                portfolio_log["date"] = pd.to_datetime(portfolio_log["date"])
                equity = (
                    portfolio_log.set_index("date")["capital"]
                    .sort_index()
                    .astype(float)
                )
                # dedup any rebalance-day + intra-stage double entries
                equity = equity[~equity.index.duplicated(keep="last")]

                self.equity_curves[(k, method)] = equity

                # AC5 — metrics computed from the equity-series path
                m = self._equity_metrics(equity)
                m.update({"K": k, "method": method})
                rows.append(m)

        self.metrics_table = pd.DataFrame(rows)
        return self

    # ------------------------------------------------------------------
    # 4. Metric block for one equity curve (paper's five rows)
    # ------------------------------------------------------------------
    def _equity_metrics(self, equity: pd.Series) -> dict:
        dr = equity_returns(equity)
        std = equity_daily_std(equity)
        sr = equity_daily_sharpe(equity)
        ann_mdd = equity_annual_max_drawdown(equity)
        ann_pnl = equity_annual_pnl(equity)

        # DailyRet / DailyStd / DailySR are point-in-time summaries — take the mean
        # (matches how paper reports a single number per column).
        daily_ret_pct = float(dr.mean() * 100.0) if not dr.empty else 0.0
        daily_std_pct = float(dr.std(ddof=1) * 100.0) if len(dr) > 1 else 0.0
        daily_sr = float(sr.iloc[-1]) if not sr.empty else 0.0

        # AnnMDD: dollar drawdown -> percentage of initial capital
        ann_mdd_dollars = float(ann_mdd.iloc[-1]) if not ann_mdd.empty else 0.0
        ann_mdd_pct = ann_mdd_dollars / self.initial_capital * 100.0

        # AnnPnL: percent of initial capital
        ann_pnl_dollars = float(ann_pnl.iloc[-1]) if not ann_pnl.empty else 0.0
        ann_pnl_pct = ann_pnl_dollars / self.initial_capital * 100.0

        return {
            "DailyRet (%)": daily_ret_pct,
            "DailyStd (%)": daily_std_pct,
            "DailySR":      daily_sr,
            "AnnMDD (%)":   ann_mdd_pct,
            "AnnPnL (%)":   ann_pnl_pct,
        }

    # ------------------------------------------------------------------
    # 5. Figure 2.4 — 2x2 grid of cumulative-return curves
    # ------------------------------------------------------------------
    def generate_figure(self, output_dir: str = "reports", filename: str = None) -> str:
        if not self.equity_curves:
            raise RuntimeError("Call .run() before .generate_figure().")

        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_figure_{ts}.png"
        out_path = os.path.join(output_dir, filename)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
        axes = axes.flatten()

        for i, k in enumerate(self.k_values):
            ax = axes[i]
            for method in self.methods:
                eq = self.equity_curves.get((k, method))
                if eq is None or eq.empty:
                    continue
                # Cumulative return relative to start (AC1)
                cum_ret = (eq / eq.iloc[0]) - 1.0
                ax.plot(
                    range(len(cum_ret)),
                    cum_ret.values,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    linewidth=1.3,
                )
            ax.set_title(f"({chr(97 + i)}) k={k}", fontsize=11)
            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative Return")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

        fig.suptitle(
            "Cumulative returns for portfolios with different allocation methods",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

        plt.show()
        fsve = input("Do you want the figure to be saved ? : ").upper()
        if fsve == 'Y':
            print(f"[ComparisonReport] Figure saved: {out_path}")

        return out_path

    # ------------------------------------------------------------------
    # 6. Table 2.5 — plaintext + CSV
    # ------------------------------------------------------------------
    def _pivoted_table(self) -> pd.DataFrame:
        """
        Wide-format DataFrame with MultiIndex(["K","metric"]) rows and
        `method` columns — matches AC4's structured output.
        """
        if self.metrics_table is None or self.metrics_table.empty:
            raise RuntimeError("Call .run() before building the table.")

        long_df = self.metrics_table.melt(
            id_vars=["K", "method"],
            var_name="metric",
            value_name="value",
        )
        pivot = long_df.pivot_table(
            index=["K", "metric"],
            columns="method",
            values="value",
        )
        # Preserve the paper's row and column ordering
        metric_order = ["DailyRet (%)", "DailyStd (%)", "DailySR", "AnnMDD (%)", "AnnPnL (%)"]
        pivot = pivot.reindex(
            pd.MultiIndex.from_product([self.k_values, metric_order], names=["K", "metric"])
        )
        pivot = pivot.reindex(columns=[m for m in self.methods if m in pivot.columns])
        return pivot

    def _best_per_row(self, row: pd.Series, metric: str) -> str:
        """
        Return the method whose value is 'best' for that metric.
        Returns/Sharpe/PnL: highest is best. Std/MDD: lowest magnitude is best
        (MDD is negative, so 'least negative' == closest to zero == max).
        """
        vals = row.dropna()
        if vals.empty:
            return None
        if metric in ("DailyRet (%)", "DailySR", "AnnPnL (%)"):
            return vals.idxmax()
        if metric == "DailyStd (%)":
            return vals.idxmin()
        if metric == "AnnMDD (%)":
            return vals.idxmax()  # negative numbers, largest == closest to 0
        return None

    def generate_table(self, highlight_best: bool = True) -> pd.DataFrame:
        """
        Print the paper's Table 2.5-style comparison to stdout only.
        Returns the pivoted DataFrame in case the caller wants it in-memory.
        """
        if self.metrics_table is None:
            raise RuntimeError("Call .run() before .generate_table().")

        pivot = self._pivoted_table()

        header_methods = list(pivot.columns)
        col_width = 12
        metric_width = 14
        total_width = metric_width + col_width * len(header_methods) + 2

        lines = []
        lines.append("=" * total_width)
        lines.append("  Comparison of allocation methods (paper Table 2.5)")
        lines.append(f"  Formation: {self.formation_start} -> {self.formation_end}")
        lines.append(f"  Trading:   {self.trading_start} -> {self.trading_end}")
        lines.append(f"  Pairs:     {len(self.pairs)}   Initial capital: ${self.initial_capital:,.0f}")
        lines.append("=" * total_width)

        for k in self.k_values:
            lines.append("")
            lines.append(f"  K = {k}")
            lines.append("  " + "-" * (metric_width + col_width * len(header_methods)))
            header = f"  {'Index':<{metric_width}}" + "".join(
                f"{m:>{col_width}}" for m in header_methods
            )
            lines.append(header)
            lines.append("  " + "-" * (metric_width + col_width * len(header_methods)))

            for metric in ["DailyRet (%)", "DailyStd (%)", "DailySR", "AnnMDD (%)", "AnnPnL (%)"]:
                key = (k, metric)
                if key not in pivot.index:
                    continue
                row = pivot.loc[key]
                best = self._best_per_row(row, metric) if highlight_best else None

                cells = []
                for m in header_methods:
                    v = row.get(m, np.nan)
                    if pd.isna(v):
                        cells.append(f"{'--':>{col_width}}")
                    else:
                        s = f"{v:.4f}"
                        if highlight_best and m == best:
                            s = f"*{s}*"
                        cells.append(f"{s:>{col_width}}")
                lines.append(f"  {metric:<{metric_width}}" + "".join(cells))

        lines.append("")
        lines.append("  * = best value in row (highest for returns/Sharpe/PnL,")
        lines.append("      lowest for Std, least-negative for MDD)")

        print("\n".join(lines))
        return pivot
    # ------------------------------------------------------------------
    # 7. Convenience one-shot
    # ------------------------------------------------------------------
    def generate_all(self, output_dir: str = "reports"):
        self.run()
        fig_path = self.generate_figure(output_dir=output_dir)
        self.generate_table()
        return {"figure": fig_path}


# ----------------------------------------------------------------------
# CLI-style entry point
# ----------------------------------------------------------------------
def main():
    config = {
        "data": {
            "formation_start": "2021-01-01",
            "formation_end":   "2021-12-31",
            "trading_start":   "2022-01-01",
            "trading_end":     "2022-12-31",
            "interval":        "1d",
        },
        "ou": {
            "hedge_ratio_grid": [-2.0, 2.0, 0.01],
            "lookback_L": 252,
        },
        "trading": {
            "M":           63,     # paper 2.4 uses M=63
            "stop_loss_r": 0.10,    # not applied when stop-loss disabled in trader
            "slippage":    0.0,    # paper has no transaction costs in ch.2
            "commission":  0.0,
        },
        "diversification": {
            "rebalance_window": 63,   # paper: quarterly
        },
    }

    pairs = [
        ["WM",   "RSG"],
        ["UAL",  "DAL"],
        ["V",    "MA"],
        ["MS",   "GS"],
        ["NVDA", "AMD"],
        ["CVX",  "XOM"],
    ]

    report = ComparisonReport(
        config=config,
        pairs=pairs,
        initial_capital=100_000.0,
        methods=["EW", "MRB", "MRR", "MVA"],
        k_values=[0.75, 1.0, 1.25, 1.5],
    )
    report.generate_all(output_dir="reports")


if __name__ == "__main__":
    main()