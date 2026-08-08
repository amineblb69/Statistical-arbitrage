import yfinance as yf
import pandas as pd 
import os 

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_pair_prices(tickers : list, start, end, itv : str):

    """ Fetch and clean pair stock data from yfinance. """

    df = pd.DataFrame()

    for tick in tickers :

        file = os.path.join(CACHE_DIR, f"{tick}_{itv}_{start}_{end}.parquet")

        if os.path.exists(file):
            stock_data = pd.read_parquet(file)

        else : 
            try :
                #Equity Data
                stock_data = yf.download(tick, start, end, interval=itv, progress=False,auto_adjust=True)
                
                if stock_data.empty:
                    raise ValueError(f"No data retrieved for ticker '{tick}'")
                
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.droplevel(1)

                stock_data.to_parquet(file)

            except Exception as e :
                raise ValueError(e)
        
        df[f'{tick}_close'] = stock_data['Close']
    df.dropna(inplace=True)
                
    return df

