import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

import strategies.diversification as _dv_module
import strategies.pairs_trading as pt
from backtest.engine import BacktestEngine


class MultiPairFramework:
    """
    Stateful, event-driven multi-pair framework suitable for LIVE TRADING
    and walk-forward backtesting.

    Paper mapping (Section 2.3.2):
        Step 1-2 : fit hedge ratios on lookback window
        Step 3   : _rebalance()  — liquidate all, recompute weights, reallocate
        Steps 4-7: on_bar()      — intra-stage trading, day by day
        Step 8   : stage_day counter triggers the next rebalance
    """

    def __init__(
        self,
        config: dict,
        pairs: List[Tuple[str, str]],
        total_capital: float,
    ):
        # --------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------
        self.config = config
        self.pairs = [tuple(p) for p in pairs]
        self.n_pairs = len(self.pairs)
        self.total_capital = total_capital

        self.method = config.get("method", "MRR")   # EW | MRB | MRR | MVA
        self.L = config.get("L", 252)              # lookback for fitting
        self.Lr = config.get("Lr", 63)             # stage length (rebalance freq)
        self.M = config.get("M", 63)               # rolling MA window
        self.K = config.get("K", 1.0)              # entry threshold
        self.r = config.get("r", 0.1)              # stop-loss fraction

        # --------------------------------------------------------------
        # Persistent state — these survive from one bar to the next
        # --------------------------------------------------------------
        self.price_history: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.column_names: Dict[Tuple[str, str], Tuple[str, str]] = {}

        self.stage_idx = -1          # -1 = not started yet
        self.stage_day = 0           # bars elapsed inside current stage

        # Cash held per pair inside the current stage
        self.pair_capital: Dict[Tuple[str, str], float] = {p: 0.0 for p in self.pairs}
        self.running_capital = total_capital

        # One trader + one engine per pair, refreshed at every rebalance
        self.pair_traders: Dict[Tuple[str, str], pt.OUpairTrader] = {}
        self.pair_engines: Dict[Tuple[str, str], BacktestEngine] = {}

        # Logs
        self.pair_daily_logs: Dict[Tuple[str, str], List[dict]] = {p: [] for p in self.pairs}
        self.portfolio_log: List[dict] = []
        self.stage_weights: List[np.ndarray] = []

        # Keep original fetch so we can restore after monkey-patching
        self._orig_fetch = getattr(_dv_module, "fetch_pair_prices", None)

    # ------------------------------------------------------------------
    # 1. History management
    # ------------------------------------------------------------------
    def ingest_history(self, historical_data: Dict[Tuple[str, str], pd.DataFrame]):
        """
        Load historical data BEFORE trading starts (formation period).
        This gives the first rebalance enough data to fit traders.
        """
        for pair, df in historical_data.items():
            self.price_history[pair] = df.copy()
            self.column_names[pair] = (df.columns[0], df.columns[1])

    def _append_bar(
        self,
        date: pd.Timestamp,
        pair: Tuple[str, str],
        s1_price: float,
        s2_price: float,
    ):
        """Add one bar to a pair's growing history."""
        s1_col, s2_col = self.column_names[pair]
        row = pd.DataFrame(
            {s1_col: [s1_price], s2_col: [s2_price]},
            index=[pd.to_datetime(date)],
        )
        if pair not in self.price_history or self.price_history[pair].empty:
            self.price_history[pair] = row
        else:
            self.price_history[pair] = pd.concat([self.price_history[pair], row])

    # ------------------------------------------------------------------
    # 2. Rebalancing = Paper Step 3  (liquidate → refit → reallocate)
    # ------------------------------------------------------------------
    def _rebalance(self, date: pd.Timestamp):
        """
        Stage transition.
        1. Liquidate every pair at today's close.
        2. Sum cash → total capital.
        3. Fit fresh OUpairTrader on the last L days.
        4. Compute diversification weights.
        5. Create fresh BacktestEngine per pair with allocated cash.
        """
        print(f"\n=== REBALANCE  Stage-{self.stage_idx + 1}  @ {date.date()} ===")

        # --- 1. Liquidate all open positions ---
        for pair in self.pairs:
            trader = self.pair_traders.get(pair)
            engine = self.pair_engines.get(pair)

            if trader is None or engine is None or trader._position == 0:
                # Nothing open; just record whatever cash the engine already holds
                self.pair_capital[pair] = engine.capital if engine else self.pair_capital.get(pair, 0.0)
                continue

            # Current raw prices
            hist = self.price_history[pair]
            if date not in hist.index:
                continue
            s1_price = float(hist.loc[date, self.column_names[pair][0]])
            s2_price = float(hist.loc[date, self.column_names[pair][1]])

            spread_price = trader.get_spread_price(date, hist.loc[:date])

            pnl = engine._execute_signal(
                signal=0,
                spread=spread_price,
                s1_price=s1_price,
                s2_price=s2_price,
                b_ratio=trader.b_ratio,
                capital=engine.capital,
                pair=list(pair),
                date=date,
            )
            engine.capital += pnl
            trader._position = 0
            trader._entry_spread = None
            self.pair_capital[pair] = engine.capital

        total_cash = sum(self.pair_capital.values())
        self.running_capital = total_cash
        print(f"Total cash after liquidation: ${total_cash:,.2f}")

        traders = {}
        for pair in self.pairs:
            df = self.price_history[pair].loc[:date]
            if len(df) > self.L:
                df = df.iloc[-self.L:]
            traders[pair] = self._fit_trader(df)
        self.pair_traders = traders

        # --- 4. Compnute weights via monkey-patched fetch ---
        def _live_fetch(pair):
            """Return only the last L days of history up to 'date'."""
            df = self.price_history[pair].loc[:date]
            if len(df) > self.L:
                df = df.iloc[-self.L:]
            return (df.iloc[:, 0].values, df.iloc[:, 1].values)

        _dv_module.fetch_pair_prices = _live_fetch
        try:
            allocator = self._create_allocator(total_cash)

            if self.method == "EW":
                allocations = allocator.compute_weights(self.n_pairs)
                allocations = np.asarray(allocations, dtype=float)
            else:
                allocations = allocator.compute_weights()
                allocations = np.asarray(allocations, dtype=float)

        finally:
            if self._orig_fetch is not None:
                _dv_module.fetch_pair_prices = self._orig_fetch

        # Safety normalisation
        if allocations.sum() < 1e-9:
            allocations = np.ones(self.n_pairs) * (total_cash / self.n_pairs)
        else:
            allocations = allocations / allocations.sum() * total_cash

        self.stage_weights.append(allocations.copy())
        print(f"Allocations: {dict(zip(self.pairs, allocations.round(2)))}")

        # --- 5. Create fresh engines with new capital ---
        for i, pair in enumerate(self.pairs):
            self.pair_capital[pair] = allocations[i]
            self.pair_engines[pair] = BacktestEngine(
                initial_capital=allocations[i],
                config=self.config,
            )

        self.stage_idx += 1
        self.stage_day = 0

    def _fit_trader(self, lookback_df: pd.DataFrame) -> pt.OUpairTrader:
        trader = pt.OUpairTrader(self.config)
        trader.fit(lookback_df)
        return trader

    def _create_allocator(self, capital: float):
        if self.method == "EW":
            return _dv_module.EqualWeightAllocator(capital)
        elif self.method == "MRB":
            return _dv_module.MeanReversionBudgetingAllocator(capital, self.pairs)
        elif self.method == "MRR":
            return _dv_module.MeanReversionRankingAllocator(capital, self.pairs)
        elif self.method == "MVA":
            return _dv_module.MeanVarianceAllocator(capital, self.pairs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    # ------------------------------------------------------------------
    # 3. Daily execution = Paper Steps 4-7
    # ------------------------------------------------------------------
    def on_bar(
        self,
        date: Union[str, pd.Timestamp],
        bar_dict: Dict[Tuple[str, str], Tuple[float, float]],
    ):
        """
        Process ONE trading bar. Call this every time new market data arrives.

        Parameters
        ----------
        date : str or pd.Timestamp
            Current trading timestamp.
        bar_dict : dict
            {("AAPL","MSFT"): (s1_price, s2_price), ...}
        """
        date = pd.to_datetime(date)

        # --- Append prices to history ---
        for pair, (s1_p, s2_p) in bar_dict.items():
            if pair not in self.column_names:
                # First time we see this pair — infer column names from config
                # or default to standard names.  If you ingested history first,
                # this block is skipped.
                self.column_names[pair] = ("s1", "s2")
            self._append_bar(date, pair, s1_p, s2_p)

        # --- Stage transition check (Paper Step 8) ---
        if self.stage_idx == -1 or self.stage_day >= self.Lr:
            self._rebalance(date)

        # --- Intra-stage trading per pair ---
        for pair in self.pairs:
            if pair not in bar_dict:
                continue

            s1_p, s2_p = bar_dict[pair]
            trader = self.pair_traders[pair]
            engine = self.pair_engines[pair]

            # History up to now (must have same columns as fit data)
            pair_hist = self.price_history[pair].loc[:date]

            # Generate signal
            signal = trader.generate_signals(date, pair_hist)

            # Spread price for the engine
            spread_price = trader.get_spread_price(date, pair_hist)

            # Execute
            pnl = engine._execute_signal(
                signal=signal,
                spread=spread_price,
                s1_price=s1_p,
                s2_price=s2_p,
                b_ratio=trader.b_ratio,
                capital=engine.capital,
                pair=list(pair),
                date=date,
            )
            engine.capital += pnl

            # Log
            self.pair_daily_logs[pair].append({
                "date": date,
                "s1": s1_p,
                "s2": s2_p,
                "signal": signal,
                "pnl": pnl,
                "capital": engine.capital,
            })

        # --- Portfolio-level snapshot ---
        total_cap = sum(
            self.pair_engines[p].capital for p in self.pairs if p in self.pair_engines
        )
        self.portfolio_log.append({
            "date": date,
            "capital": total_cap,
            "stage": self.stage_idx,
            "stage_day": self.stage_day,
        })

        self.stage_day += 1

    # ------------------------------------------------------------------
    # 4. Backtest convenience (walk-forward, bar-by-bar)
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        price_data: Dict[Tuple[str, str], pd.DataFrame],
        trading_start: Union[str, pd.Timestamp],
        trading_end: Union[str, pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Feed historical data bar-by-bar through on_bar().
        This is NOT a vectorised backtest — it is a walk-forward simulation
        that uses exactly the same code path as live trading.
        """
        # Ingest everything so history is available before the first bar
        self.ingest_history(price_data)

        # Determine common trading dates
        first_pair = self.pairs[0]
        df = price_data[first_pair]
        if trading_start:
            df = df.loc[pd.to_datetime(trading_start):]
        if trading_end:
            df = df.loc[:pd.to_datetime(trading_end)]

        dates = df.index

        for date in dates:
            bar_dict = {}
            for pair, ddf in price_data.items():
                if date in ddf.index:
                    row = ddf.loc[date]
                    bar_dict[pair] = (float(row.iloc[0]), float(row.iloc[1]))
            if bar_dict:
                self.on_bar(date, bar_dict)

        return pd.DataFrame(self.portfolio_log)

   
    def get_portfolio_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_log)

    def get_pair_curve(self, pair: Tuple[str, str]) -> pd.DataFrame:
        return pd.DataFrame(self.pair_daily_logs.get(pair, []))

    def get_weights_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.stage_weights,
            columns=[f"{p[0]}-{p[1]}" for p in self.pairs],
        )