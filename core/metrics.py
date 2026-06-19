import pandas as pd
import numpy as np



# In metrics.py, replace daily_return with this:

def daily_return(trades_df):
    """
    Daily return from trade log. Uses exit_date as the PnL realization date.
    """
    if trades_df.empty:
        return pd.Series(dtype=float).rename('daily_return')
    
    # Use exit_date as the date when PnL is realized
    daily = trades_df.groupby('exit_date')['net_pnl'].sum()
    equity = daily.cumsum()
    dr = equity.pct_change().dropna().rename('daily_return')

    return dr

def daily_std(pnl_series):
    """Rolling (expanding) std dev of daily returns, as a series."""
    
    dr = daily_return(pnl_series)
    ds = dr.expanding().std(ddof=1).rename('daily_std')
    
    return ds

def daily_sharpe(pnl_series, rf=0.0, ann=252):
    """Expanding annualised Sharpe on daily returns, as a series."""
    
    dr        = daily_return(pnl_series)
    excess    = dr - rf / ann
    roll_mean = excess.expanding().mean()
    roll_std  = excess.expanding().std(ddof=1)
    shrp      = (np.sqrt(ann) * roll_mean / roll_std).rename('daily_sharpe')
    
    return shrp

def annual_max_drawdown(trades_df):
    """Max drawdown on equity curve, computed per calendar year."""
    if trades_df.empty:
        return pd.Series(dtype=float).rename('annual_max_drawdown')
    
    daily = trades_df.groupby('exit_date')['net_pnl'].sum()
    equity = daily.cumsum()
    equity.index = pd.to_datetime(equity.index)

    def _mdd(s):
        roll_max = s.cummax()
        return (s - roll_max).min()

    return equity.groupby(equity.index.year).apply(_mdd).rename('annual_max_drawdown')

def annual_pnl(trades_df):
    """Total PnL summed per calendar year."""
    if trades_df.empty:
        return pd.Series(dtype=float).rename('annual_pnl')
    
    daily = trades_df.groupby('exit_date')['net_pnl'].sum()
    daily.index = pd.to_datetime(daily.index)

    return daily.groupby(daily.index.year).sum().rename('annual_pnl')

def performance_summary(pnl_series, rf=0.0, ann=252):
    dr = daily_return(pnl_series)
    return {
        "avg_daily_return":  (dr.mean() * 100),
        "avg_daily_std":     (dr.std(ddof=1)*100),
        "daily_sharpe":      (np.sqrt(ann) * dr.mean() / dr.std(ddof=1)),
        "annual_pnl":        (annual_pnl(pnl_series).sum() * 100),
        "annual_max_dd":     (annual_max_drawdown(pnl_series).min() * 100),
    }


# ─────────────────────────────────────────────────────────────────────────
# EQUITY-CURVE-BASED METRICS
#
# Everything above this point expects a *trade log* (a DataFrame with
# 'exit_date' and 'net_pnl' columns) and internally derives an equity curve
# via groupby('exit_date').cumsum(). That works for a single pair's trade
# log, but it doesn't fit data that is *already* a daily equity/capital
# series — e.g. MultiPairFramework.portfolio_log, which has one row per
# trading bar (date -> capital), not one row per closed trade.
#
# Rather than force that data through the trade-log shape, the functions
# below take an equity series directly (pd.Series, date-indexed, in
# dollars) and apply the *same formulas* as above. This lets portfolio-level
# code reuse the math without reusing the trade-log parsing.
#
# Relationship to the functions above:
#   daily_return(trades_df)        ~  groupby+cumsum  -> equity_returns(equity)
#   daily_std(trades_df)           ~  equity_daily_std(equity)
#   daily_sharpe(trades_df)        ~  equity_daily_sharpe(equity)
#   annual_max_drawdown(trades_df) ~  equity_annual_max_drawdown(equity)
#   annual_pnl(trades_df)          ~  equity_annual_pnl(equity)
#   compute_metrics(trades_df,...) ~  compute_metrics_from_equity(equity,...)
# ─────────────────────────────────────────────────────────────────────────

def equity_returns(equity):
    """Daily simple returns from an equity/capital series. E_t/E_{t-1} - 1."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float).rename('daily_return')

    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    return eq.pct_change().dropna().rename('daily_return')


def equity_daily_std(equity):
    """Expanding std dev of daily returns derived from an equity series."""
    dr = equity_returns(equity)
    return dr.expanding().std(ddof=1).rename('daily_std')


def equity_daily_sharpe(equity, rf=0.0, ann=252):
    """Expanding annualised Sharpe derived from an equity series."""
    dr        = equity_returns(equity)
    excess    = dr - rf / ann
    roll_mean = excess.expanding().mean()
    roll_std  = excess.expanding().std(ddof=1)
    return (np.sqrt(ann) * roll_mean / roll_std).rename('daily_sharpe')


def equity_drawdown(equity):
    """Dollar drawdown series: E_t - running_max(E_t)."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float).rename('drawdown')

    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()
    roll_max = eq.cummax()

    return (eq - roll_max).rename('drawdown')


def equity_annual_max_drawdown(equity):
    """Max drawdown computed per calendar year from an equity series."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float).rename('annual_max_drawdown')

    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    def _mdd(s):
        roll_max = s.cummax()
        return (s - roll_max).min()

    return eq.groupby(eq.index.year).apply(_mdd).rename('annual_max_drawdown')


def equity_annual_pnl(equity):
    """
    Annual PnL implied by an equity series: last value of the year minus
    last value of the prior year (first year measured from the series start).
    """
    if equity is None or equity.empty:
        return pd.Series(dtype=float).rename('annual_pnl')

    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    year_end = eq.groupby(eq.index.year).last()
    prior_close = year_end.shift(1)
    prior_close.iloc[0] = eq.iloc[0]  # baseline for the first year
    return (year_end - prior_close).rename('annual_pnl')


def compute_metrics_from_equity(equity, initial_capital, rf=0.0, ann=252):
    """
    Compute the same family of summary metrics as compute_metrics(), but
    starting from a daily equity/capital series instead of a trade log.

    Use this for anything that isn't naturally a per-trade log — e.g. a
    multi-pair portfolio capital curve (MultiPairFramework.portfolio_log).

    Note: trade-count statistics (win rate, profit factor, avg win/loss,
    etc.) have no meaning here since there's no trade log — those are
    simply omitted. If you also have a trade log available (e.g. by
    concatenating every pair's trade log), compute those separately with
    compute_metrics() and merge the dicts.
    """
    if equity is None or equity.empty:
        return {
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "avg_daily_return": 0.0,
            "avg_daily_std": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_pnl": {},
            "annual_max_dd": {},
            "calmar_ratio": 0.0,
        }

    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    total_pnl = eq.iloc[-1] - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100

    dr = equity_returns(eq)
    avg_daily_return = dr.mean() * 100 if not dr.empty else 0.0
    avg_daily_std = dr.std(ddof=1) * 100 if not dr.empty else 0.0
    dstd = dr.std(ddof=1) if not dr.empty else 0.0
    sharpe_ratio = (
        np.sqrt(ann) * dr.mean() / dstd
        if not dr.empty and dstd not in (0, None) and not np.isnan(dstd)
        else 0.0
    )

    roll_max = eq.cummax()
    drawdown = eq - roll_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (
        (max_drawdown / roll_max.loc[drawdown.idxmin()]) * 100
        if max_drawdown != 0 else 0.0
    )

    ann_pnl = equity_annual_pnl(eq)
    ann_mdd = equity_annual_max_drawdown(eq)

    avg_annual_return = ann_pnl.mean() if not ann_pnl.empty else 0.0
    calmar_ratio = avg_annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "avg_daily_return": avg_daily_return,
        "avg_daily_std": avg_daily_std,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "annual_pnl": ann_pnl.to_dict(),
        "annual_max_dd": ann_mdd.to_dict(),
        "calmar_ratio": calmar_ratio,
    }


def compute_metrics(trades_df, results_df, initial_capital, rf=0.0, ann=252):
    """
    Compute all key performance metrics from trade log and daily results.
    """
    if trades_df.empty:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_trade_pnl": 0.0,
            "min_trade_pnl": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "avg_daily_return": 0.0,
            "avg_daily_std": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_pnl": {},
            "annual_max_dd": {},
            "calmar_ratio": 0.0,
        }

    # --- Trade-level metrics ---
    total_trades = len(trades_df)
    winning_trades = int((trades_df['net_pnl'] > 0).sum())
    losing_trades = int((trades_df['net_pnl'] < 0).sum())
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    avg_trade_pnl = trades_df['net_pnl'].mean()
    avg_win = trades_df.loc[trades_df['net_pnl'] > 0, 'net_pnl'].mean() if winning_trades > 0 else 0.0
    avg_loss = trades_df.loc[trades_df['net_pnl'] < 0, 'net_pnl'].mean() if losing_trades > 0 else 0.0

    gross_profit = trades_df.loc[trades_df['net_pnl'] > 0, 'net_pnl'].sum()
    gross_loss = abs(trades_df.loc[trades_df['net_pnl'] < 0, 'net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    max_trade_pnl = trades_df['net_pnl'].max()
    min_trade_pnl = trades_df['net_pnl'].min()

    total_pnl = trades_df['net_pnl'].sum()
    total_return_pct = (total_pnl / initial_capital) * 100

    # --- Daily metrics ---
    dr = daily_return(trades_df)
    avg_daily_return = dr.mean() * 100
    avg_daily_std = dr.std(ddof=1) * 100
    sharpe_ratio = np.sqrt(ann) * dr.mean() / dr.std(ddof=1) if dr.std(ddof=1) != 0 else 0.0

    # --- Drawdown ---
    daily = trades_df.groupby('exit_date')['net_pnl'].sum()
    equity = daily.cumsum() + initial_capital
    equity.index = pd.to_datetime(equity.index)
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / roll_max.loc[drawdown.idxmin()]) * 100 if drawdown.min() != 0 else 0.0

    # --- Annual metrics ---
    ann_pnl = annual_pnl(trades_df)
    ann_mdd = annual_max_drawdown(trades_df)

    # Calmar = annual return / abs(max drawdown)
    avg_annual_return = ann_pnl.mean() if not ann_pnl.empty else 0.0
    calmar_ratio = avg_annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate * 100,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_trade_pnl": max_trade_pnl,
        "min_trade_pnl": min_trade_pnl,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "avg_daily_return": avg_daily_return,
        "avg_daily_std": avg_daily_std,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "annual_pnl": ann_pnl.to_dict(),
        "annual_max_dd": ann_mdd.to_dict(),
        "calmar_ratio": calmar_ratio,
    }