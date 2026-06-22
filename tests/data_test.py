import yfinance as yf

print(yf.download(tickers='MS',start="2022-12-31",end="2023-12-31",interval='1d'))