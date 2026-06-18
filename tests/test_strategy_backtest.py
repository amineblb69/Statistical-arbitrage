import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import os
import pandas as pd
from core.metrics import performance_summary
from data.loader import fetch_pair_prices
from strategies.pairs_trading import OUpairTrader
from backtest.engine import BacktestEngine

def main():
    
    config = {
        "trading": {
            "M": 30,               
            "K": 1.0,             
            "stop_loss_r": 0.15,    
            "slippage": 0.0005,    
            "commission": 0.001,   
        },
        "ou": {
            "hedge_ratio_grid": [0.1, 2.0, 0.01] 
        }
    }

    
    tickers = ["NVDA", "AMD"] 
    formation_start = "2023-01-01"
    formation_end = "2023-12-31"
    trading_end = "2025-12-31"
    interval = "1d"

    print(f"Fetching data for pair: {tickers}...")
    
    full_df = fetch_pair_prices(tickers, start=formation_start, end=trading_end, itv=interval)
    
    formation_data = full_df.loc[formation_start:formation_end].copy()
    trading_data = full_df.loc[formation_end:trading_end].copy()

    if formation_data.empty or trading_data.empty:
        print("Error: Retrieved dataframes are empty. Check your dates or internet connection.")
        return

    print(f"Formation period rows: {len(formation_data)}")
    print(f"Trading period rows: {len(trading_data)}")

    # 3. Instantiate and fit the strategy
    print("\nFitting Ornstein-Uhlenbeck parameters on formation data...")
    strategy = OUpairTrader(config)
    strategy.fit(formation_data)
    
    print(f"--- Fit Results ---")
    print(f"Optimal Hedge Ratio (B): {strategy.b_ratio}")
    print(f"Mean reversion level (mu): {strategy.mu}")
    print(f"Speed of reversion (theta): {strategy.theta}")
    print(f"Volatility (sigma): {strategy.sigma}")

    # 4. Initialize the backtest engine
    initial_capital = 100_000.0
    print(f"\nInitializing engine with ${initial_capital:,.2f}...")
    engine = BacktestEngine(initial_capital=initial_capital, config=config)

    # 5. Run the backtest
    print("Running simulation...")
    results_df = engine.run(pair_id=tickers, strategy=strategy, price_data=trading_data)

    # 6. Display basic results
    print("\n--- Backtest Summary ---")
    final_capital = results_df['capital'].iloc[-1]
    total_pnl = final_capital - initial_capital
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Net PnL: ${total_pnl:,.2f} ({(total_pnl / initial_capital) * 100:.2f}%)")
    print(f"Total Trades Executed: {len(engine.trade_log)}")

    # Optional: Preview trade logs if any occurred
    if engine.trade_log:
        trades_df = pd.DataFrame(engine.trade_log)
        print("\nAll trades executed lavamille")
        print(trades_df[['side', 'entry_date', 'exit_date', 'net_pnl','return']])

    #Diplay Metrics Summary

    ''' perf_results = performance_summary(trades_df)

    print("\n")
    print(5*"="+"PERF RESUME : "+5*"=")
    print("Average Daily Return : ", perf_results['avg_daily_return'])
    print("Daily Return : ", perf_results['avg_daily_return'])'''


if __name__ == "__main__":
    main()

