import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import loader
from core.spread import construct_spread
from core.ou_fitting import fit_hedge_ratio

data = loader.fetch_pair_prices(['GS','MS'],'2021-01-01','2023-01-01','1d')

s1 = data['GS_close']
s2 = data['MS_close']

b_r, mu, theta , sigma = fit_hedge_ratio(s1,s2)
M = 63
spr = construct_spread(s1,s2,b_r)

spread_window = spr.iloc[-M:]  
ma  = spr.rolling(M).mean()
std = spr.rolling(M).std()

print(spr)
print(f"THE HEDGE RATIO CALCULATED BY OU FITTING IS : {b_r} \n WITH THE FOLLOWING PARAMS : mu({mu}), theta({theta}), sigma({sigma})")
print(spread_window,ma,std)