from data import loader
from core.spread import construct_spread
from core.ou_fitting import fit_hedge_ratio

data = loader.fetch_pair_prices(['GS','MS'],'2025-01-01','2026-01-01','1d')

s1 = data['GS_close']
s2 = data['MS_close']

b_r, mu, theta , sigma = fit_hedge_ratio(s1,s2)

spr = construct_spread(s1,s2,b_r)
print(spr)
print("THE HEDGE RATIO CALCULATED BY OU FITTING IS : {b_r} \n WITH THE FOLLOWING PARAMS : mu({mu}), theta({theta}), sigma({sigma})")