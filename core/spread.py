import pandas as pd
import numpy as np

def construct_spread(s1: pd.Series, s2: pd.Series, B: float):

    s1 = s1 / s1[0]
    s2 = s2 / s2[0]
    spread = s1 - B*s2


    return spread

def rolling_mean(spread , M):

    rl_mean = spread.rolling(window=M).mean()

    return rl_mean

def rolling_std(spread, M):

    rl_std = spread.rolling(window = M).std(ddof=1)

    return rl_std

def simulate_ou(mu, theta, sigma, T, dt, n_paths, x0 = None):
    
    n_steps = int(T/dt)
    x0      = theta if x0 is None else x0    
    paths   = np.zeros((n_paths, n_steps + 1))
    paths[:,0] = x0

    for t in range(1, n_steps+1):
        dW =  np.random.normal(0, np.sqrt(dt), size = n_paths)
        paths[:,t] = (paths[:, t-1] + 
                      mu * (theta - paths[:, t-1]) * dt 
                      + sigma * dW)
        
    return paths