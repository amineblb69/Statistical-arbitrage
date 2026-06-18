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

