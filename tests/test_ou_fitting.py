from data.loader import fetch_pair_prices
from core.spread import construct_spread
from core.ou_fitting import fit_hedge_ratio

pair = ['NVDA','AMD']

data = fetch_pair_prices(pair,'2024-01-01','2026-01-01','1d')

s1 = data[f'{pair[0]}_close']
s2 = data[f'{pair[1]}_close']

b_r, mu, theta , sigma = fit_hedge_ratio(s1,s2)

spr = construct_spread(s1,s2,b_r)
print(spr)
print(f"THE HEDGE RATIO CALCULATED BY OU FITTING IS : {b_r} \n WITH THE FOLLOWING PARAMS : mu({mu}), theta({theta}), sigma({sigma})")