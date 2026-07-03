import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from data.loader import fetch_pair_prices
from core.spread import construct_spread
from core.ou_fitting import fit_hedge_ratio, compute_log_likelihood

pair = ['NVDA','AMD']

data = fetch_pair_prices(pair,'2021-01-01','2021-12-31','1d')

s1 = data[f'{pair[0]}_close']
s2 = data[f'{pair[1]}_close']

b_r, mu, theta , sigma = fit_hedge_ratio(s1,s2)

spr = construct_spread(s1,s2,b_r)
ll  = compute_log_likelihood(spr, mu, theta, sigma)

print(spr)
print(f"PAIR : {pair}")
print(f"THE HEDGE RATIO CALCULATED BY OU FITTING IS : {b_r} \n WITH THE FOLLOWING PARAMS : mu({mu}), theta({theta}), sigma({sigma})")
print(f"WE COMPUTE THE FOLLOWING LOG LIKELIHOOD SCORE : {ll:.2f}")