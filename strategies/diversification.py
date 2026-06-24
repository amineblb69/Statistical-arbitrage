import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from core.ou_fitting import compute_log_likelihood, fit_hedge_ratio
from core.spread import construct_spread
from data.loader import fetch_pair_prices

class EqualWeightAllocator:

    def __init__(self, capital):
        self.capital = capital
        
    
    def compute_weights(self, n_pairs):
        """
        This function allocate all the available cash equally to the n pair traded.
        """

        cash_allocation = self.capital / n_pairs
        weights         = [cash_allocation] * n_pairs

        self.capital_weights = pd.array(weights)

        return self.capital_weights
    
class MeanReversionBudgetingAllocator:

    def __init__(self, capital, pairs):
        self.capital = capital
        self.pairs = pairs

        self.b_ratios = []
        self.mus      = []
        self.thetas   = []
        self.sigmas    = []
        self.likelihoods = []

        self.l_tilde = None
        self.mu_tilde = None
        self.raw_weights = None
        self.capital_weights = None

    def compute_params(self):

        self.b_ratios.clear()
        self.mus.clear()
        self.thetas.clear()
        self.sigmas.clear()
        self.likelihoods.clear()

        for pair in self.pairs:
            
            pair_data = fetch_pair_prices(pair)
            s1        = pair_data[0]
            s2        = pair_data[1]
        
            b_ratio, mu, theta, sigma = fit_hedge_ratio(s1, s2)
            spread  = construct_spread(s1,s2,b_ratio) 
            ll      = compute_log_likelihood(spread,mu,theta,sigma)
            
            self.b_ratios.append(b_ratio)
            self.mus.append(mu)
            self.thetas.append(theta)
            self.sigmas.append(sigma)
            self.likelihoods.append(ll)

        return self
    
    @staticmethod
    def _min_max_normalize(values):
        """
        Min-max normalize an array to [0, 1].
        If all values are identical, returns an array of ones.
        """
        arr  = np.asarray(values, dtype=float)
        vmin = arr.min()
        vmax = arr.max()

        if vmax - vmin < 1e-12:
            return np.ones_like(arr)
        return (arr - vmin) / (vmax - vmin)
    
    def compute_weights(self):


        if not self.likelihoods:
            self.compute_params()
        n = len(self.pairs)

        #Compute Ltilde : (l^i − min{ˆ li} )/( max{ˆ li} − min{ˆ li})
        self.l_tilde = self._min_max_normalize(self.likelihoods)
        
        #Compute Scaled Mean Reversion Mean : mu_star_i = mu_i / sigma_i
        mus        = np.array(self.mus, dtype=float)
        sigmas     = np.array(self.sigmas, dtype=float)
        sigma_safe = np.where(np.abs(sigmas) < 1e-12, 1e-12, sigmas)
        mu_star    = mus / sigma_safe

        
        # Normative relative speed : mu_tilde_i = (mu_star_i - min(mu_star)) / (max(mu_star) - min(mu_star))
        self.mu_tilde = self._min_max_normalize(mu_star)

        combine = self.l_tilde * self.mu_tilde
        total   = combine.sum()

        if total < 1e-12:
            # Fallback to equal weighting if all scores are zero
            self.raw_weights = np.ones(n) / n
        else:
            self.raw_weights = combine / total

        self.capital_weights = pd.array(self.raw_weights * self.capital)

        return self.capital_weights

class MeanReversionRankingAllocator:

    def __init__(self, capital, pairs):
        self.capital = capital
        self.n       = len(pairs)
        self.pairs   = pairs

        # Parallel lists — index i always refers to self.pairs[i]
        self.b_ratios    = []
        self.mus         = []
        self.thetas      = []
        self.sigmas      = []
        self.likelihoods = []

        self.l_tilde  = None
        self.mu_tilde = None
        self.capital_weights = None

    def compute_params(self):

        self.b_ratios.clear()
        self.mus.clear()
        self.thetas.clear()
        self.sigmas.clear()
        self.likelihoods.clear()

        for pair in self.pairs:
            
            pair_data = fetch_pair_prices(pair)
            s1        = pair_data[0]
            s2        = pair_data[1]
        
            b_ratio, mu, theta, sigma = fit_hedge_ratio(s1, s2)
            spread  = construct_spread(s1,s2,b_ratio) 
            ll      = compute_log_likelihood(spread,mu,theta,sigma)
            
            self.b_ratios.append(b_ratio)
            self.mus.append(mu)
            self.thetas.append(theta)
            self.sigmas.append(sigma)
            self.likelihoods.append(ll)
        
        mu_star = [m / s if abs(s) > 1e-12 else 0.0
                   for m, s in zip(self.mus, self.sigmas)]
        self.mu_tilde = self._min_max_normalize(mu_star)

        

        return self
    
    @staticmethod
    def _min_max_normalize(values):

        arr = np.asarray(values, dtype=float)
        vmin, vmax = arr.min(), arr.max()

        if vmax - vmin < 1e-12:
            return np.ones_like(arr)
        
        return (arr - vmin) / (vmax - vmin)
    
    def compute_weights(self):
        if self.l_tilde is None:
            self.compute_params()

        # Edge case: single pair
        if self.n == 1:
            self.capital_weights = pd.array([self.capital])
            return self.capital_weights

        products = [self.l_tilde[i] * self.mu_tilde[i] for i in range(self.n)]

        ranked_indices = np.argsort(products)

        raw_weights = np.zeros(self.n)

        for rank_pos, pair_idx in enumerate(ranked_indices):
            raw_weights[pair_idx] = (self.n - 1 + 2 * rank_pos) / (2 * self.n * (self.n - 1))

        self.capital_weights = pd.array(raw_weights * self.capital)


        return self.capital_weights

class MeanVarianceAllocator:

    def __init__(self, capital, pairs):
        self.capital = capital
        self.pairs   = pairs
        self.n       = len(pairs)

        # Per-pair daily return series (list of 1-D arrays)
        self.returns = []

        self.raw_weights     = None
        self.capital_weights = None

    def compute_params(self):
        """
        Fetch prices, fit hedge ratio, and compute daily returns for each pair.
        """
        self.returns = []

        for pair in self.pairs:
            pair_data = fetch_pair_prices(pair)
            s1 = np.asarray(pair_data[0], dtype=float)
            s2 = np.asarray(pair_data[1], dtype=float)

            # Fit hedge ratio to know the portfolio composition
            b_ratio, _, _, _ = fit_hedge_ratio(s1, s2)

            # Daily simple returns of the individual assets
            r1 = np.diff(s1) / s1[:-1]
            r2 = np.diff(s2) / s2[:-1]

            # Pair strategy return: long 1 unit of s1, short b_ratio units of s2
            pair_ret = r1 - b_ratio * r2

            self.returns.append(pair_ret)

        return self

    def compute_weights(self):
        """
        Solve:  max_w (w^T R) / (w^T Σ w)
                s.t.  ||w||_1 = 1  ,  w >= 0
        """
        if not self.returns:
            self.compute_params()

        # Align to common lookback length
        min_len = min(len(r) for r in self.returns)
        if min_len < 2:
            raise ValueError("MVA requires at least 2 daily return observations per pair.")

        # T x N matrix of daily returns
        R_mat = np.column_stack([r[:min_len] for r in self.returns])

        # Mean return vector and covariance matrix
        R     = np.mean(R_mat, axis=0)
        Sigma = np.cov(R_mat, rowvar=False)

        # Regularize to avoid singular covariance
        if self.n == 1:
            Sigma = np.array([[np.var(R_mat[:, 0])]])
        else:
            Sigma = Sigma + 1e-6 * np.eye(self.n)

        # --- Optimization (paper's exact objective) ---
        def neg_objective(w):
            num = w @ R
            den = w @ Sigma @ w
            if den <= 1e-12:
                return 0.0 if num <= 0 else -1e12
            return -(num / den)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds      = [(0.0, 1.0) for _ in range(self.n)]
        w0          = np.ones(self.n) / self.n

        result = minimize(
            neg_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if result.success:
            weights = result.x
        else:
            # Fallback to equal weights if optimizer fails
            weights = np.ones(self.n) / self.n

        # Clean numerical noise and re-normalize
        weights = np.maximum(weights, 0.0)
        weights = weights / np.sum(weights)

        self.raw_weights     = weights
        self.capital_weights = pd.array(weights * self.capital)

        return self.capital_weights



        

        





     
        
