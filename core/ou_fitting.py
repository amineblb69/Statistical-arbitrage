from core.spread import construct_spread
import numpy as np

def compute_ou_params(series,dt=1):
    series = np.array(series)

    x = series[:-1]
    y = series[1:]
    n = len(series) - 1
    
    Xx  = np.sum(x)
    Xy  = np.sum(y)
    Xxx = np.sum(x**2)
    Xyy = np.sum(y**2)
    Xxy = np.sum(x*y)

    #Theta Calculation 
    nom_theta = (Xy * Xxx) - (Xx * Xxy)
    den_theta = n * (Xxx - Xxy) - (Xx**2 - Xx*Xy)
    theta     = nom_theta / den_theta

    #Mu Mean Reversion Speed Calculation 
    nom_mu = Xxy - theta*Xx - theta*Xy + n* theta**2
    den_mu = Xxx - 2*theta*Xx + n* theta**2
    mu     = -(1/dt) * np.log(nom_mu/den_mu)

    #Sigma Volatility Calculation
    alpha  = np.exp(-mu * dt)
    alpha2 = np.exp(-2 * mu * dt)
    numerator = (
        Xyy - 2 * alpha  * Xxy
        + alpha2 * Xxx
        - 2 * theta  * (1- alpha) * (Xy - alpha * Xx)
        + n * theta**2 * (1- alpha)**2
    )
    var   = (2 * mu / (n * (1- alpha2))) * numerator
    sigma = np.sqrt(max(var, 1e-12))

    return mu, theta, sigma

def compute_log_likelihood(series, mu, theta, sigma, dt=1):

    #Raw inputs
    x    = series[:-1]
    y    = series[1:]
    l    = len(y)

    #
    alpha      = np.exp(-mu*dt)
    sigma_t_sq = (sigma**2) * (1 - alpha**2) / (2*mu)
    sigma_t    = np.sqrt(sigma_t_sq) 
    residuals  = y - x * alpha - theta * (1 - alpha)

    #Log Likelihood Computation
    ll = (
        -0.5 * np.log(2 * np.pi)
        - np.log(sigma_t)
        - (1 / (2 * sigma_t_sq)) * np.mean(residuals**2)
    )

    return ll

def fit_hedge_ratio(s1, s2, grid_min=-2, grid_max =2, grid_step = 0.01):

    """ Fit the spread hedge ratio by maximizing likelihood estimation function """

    s1 = np.array(s1)
    s2 = np.array(s2)

    b_values    = np.arange(grid_min,grid_max,grid_step)
    best_b      = None    
    best_ll     = -np.inf
    best_params = (None,None,None)
    
    #Loop to grid search 
    for b in b_values:

        spread = construct_spread(s1,s2,b)

        try :

            mu, theta, sigma = compute_ou_params(spread)

            if mu <= 0 or sigma <=0 :
                continue

            ll               = compute_log_likelihood(spread, mu, theta, sigma) 

            if ll > best_ll :

                best_ll     = ll
                best_b      = b
                best_params = mu, theta, sigma 

        except Exception as e :
            print("Problem fitting the hedge ratio b({b}): ",e)
            
    mu_ , theta_, sigma_ = best_params
    
    return best_b, mu_, theta_, sigma_