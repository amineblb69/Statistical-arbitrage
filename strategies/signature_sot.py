"""
strategies/signature_sot.py
---------------------------
Signature Optimal Stopping (SOT).
 
Given:
  - M training sample paths X^(m)_t (1D, e.g. simulated spreads),
  - The associated payoffs Y^(m)_t (e.g. e^{-rt}(-X_t - c) for a long-entry),
  - A truncation level N and threshold k,
 
we learn the linear stopping policy l* that minimizes the loss (Eq. 3.19)
and then apply the deterministic rule (Eq. 3.18) to decide when to stop
on a new path.
 
Loss (Eq. 3.19):
    L(l) = - (1/M) Σ_m [ Y_0^(m)
                       + Σ_{j=0..n-1} Ĝ_k( Σ_{i≤j} ⟨l, sig^(m)_i⟩² )
                                        · (Y^(m)_{j+1} − Y^(m)_j) ]
 
with Ĝ_k(x) = 1 − σ(μ(x−k))  (smoothed complementary CDF of a constant Z=k).
 
Stopping rule (Eq. 3.18):
    ν = inf { j : Σ_{i≤j} ⟨l*, sig_i⟩² ≥ k }
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
import numpy as np
from scipy.optimize import minimize
 
from core.signatures import (
    time_augment,
    path_signature_stream,
    sig_length,
)
 

class SignatureOptimalStopping : 


    """
    Fit-then-predict object for the signature optimal stopping problem.
 
    Parameters
    ----------
    N : int
        Truncation level of the signature (paper: usually 3 or 4).
    k : float
        Deterministic threshold in Eq. (3.18); replaces the random variable Z.
    mu : float
        Sharpness of the sigmoid used to smooth the indicator (paper: 20).
    T : float
        Path horizon used for time-augmentation. Paths are assumed to be
        sampled on an equidistant grid over [0, T].
    """

    def __init__(self, N:int = 3, k:float = 1.0, mu:float=2, T:float = 1.0):

        self.N = N
        self.k = k
        self.mu = mu
        self.T = T

        self.l_star = None #shape (sig_length(1+d, N),)
        self.d      = None #dimension of the raw path (1 for a spread)


    #Signature Pe-computation

    def _stream(self, x:np.ndarray) -> np.ndarray :

        aug    = time_augment(x, T=self.T)
        stream = path_signature_stream(aug, N=self.N)

        return np.stack(stream)
    
    #Loss 

    def _loss(self, l_vec:np.ndarray, sigs:np.ndarray, payoffs:np.ndarray) -> float:
        """
        sigs : (M,n+1,L) augmented signature stream per training path 
        payoffs : (M, n+1) payoff process per training path
        """

        M, n1, L = sigs.shape
        n = n1 - 1

        # ⟨l, sig^(m)_i⟩² for every m, i  →  shape (M, n+1)
        inner    = sigs @ l_vec
        inner_sq = inner * inner 

        #Cumulative sum Σ_{i≤j} ⟨l, sig^(m)_i⟩²  →  shape (M, n+1)
        cum = np.cumsum(inner_sq, axis = 1)

        # Ĝ_k(x) = 1 − σ(μ(x − k)) = σ(−μ(x − k))
        # (using the σ(-z) = 1 - σ(z) identity avoids one subtraction)
        Ghat = _sigmoid(-self.mu * (cum - self.k))      
        
        # (M, n+1) We only need G_hat evaluated at j = 0..n-1 to multiply (Y_{j+1} - Y_j)
        dY = np.diff(payoffs, axis=1)                   # (M, n)
        weighted = Ghat[:, :n] * dY                     # (M, n)
 
        # Expected payoff estimate per path: Y_0 + Σ_j Ĝ · dY
        per_path = payoffs[:, 0] + weighted.sum(axis=1) # (M,)
 
        # We MAXIMIZE expectation → MINIMIZE its negative mean
        return -per_path.mean()
    
    #Training

    def fit(self, paths: np.ndarray, payoffs: np.ndarray, verbose: bool = False):
        """
        paths   : (M, n+1) array of 1D training paths (rows = samples).
        payoffs : (M, n+1) array of matching payoff processes Y^(m)_t.
        """
        paths   = np.asarray(paths,   dtype=float)
        payoffs = np.asarray(payoffs, dtype=float)
        assert paths.shape == payoffs.shape, "paths and payoffs must align"
 
        self.d = 1
        L = sig_length(1 + self.d, self.N)              # augmented dim = 1+d
 
        # Pre-compute signature streams for every training sample once.
        # This is the expensive part; the optimizer will only do dot products.
        sigs = np.stack([self._stream(x) for x in paths], axis=0)  # (M, n+1, L)

      # ---- Normalize signature features (per-coordinate std) --------------
        # Signature levels grow like (path length)^k / k!, so higher levels have
        # very different scales than level 1. We standardize so ⟨l, sig⟩ is
        # meaningful for a common learning rate.
        scales = sigs.reshape(-1, L).std(axis=0)
        scales[scales < 1e-10] = 1.0                    # keep the level-0 "1"
        self._sig_scales = scales
        sigs = sigs / scales                            # broadcast over (M, n+1)
 
        # ---- Initial l chosen large enough to break the flat region ---------
        # We want the cumulative score to be O(k) somewhere in the path,
        # otherwise Ĝ_k ≈ 1 everywhere and the loss is flat in l.
        rng = np.random.default_rng(0)
        l0 = 0.5 * rng.standard_normal(L) / np.sqrt(L)
 
        result = minimize(
            fun=self._loss,
            x0=l0,
            args=(sigs, payoffs),
            method="L-BFGS-B",
            options={"maxiter": 500, "disp": verbose},
        )
 
        self.l_star = result.x
        self._last_result = result
        return self
 
    # ------------------------------------------------------------------
    # Prediction (Eq. 3.18)
    # ------------------------------------------------------------------
    def stopping_time(self, x: np.ndarray) -> int:
        """
        Deterministic stopping rule applied to a new observed path `x`
        (1D array of length n+1). Returns the first index j such that
                Σ_{i≤j} ⟨l*, sig_i⟩² ≥ k
        or n (last index) if the threshold is never crossed.
        """
        assert self.l_star is not None, "call .fit() first"
        sigs = self._stream(x) / self._sig_scales       # (n+1, L)
        inner_sq = (sigs @ self.l_star) ** 2            # (n+1,)
        cum = np.cumsum(inner_sq)                       # (n+1,)
        crossed = np.where(cum >= self.k)[0]
        return int(crossed[0]) if crossed.size else len(x) - 1
 
    def stopping_score(self, x: np.ndarray) -> np.ndarray:
        """Diagnostic: returns the cumulative score Σ_{i≤j} ⟨l*, sig_i⟩²."""
        sigs = self._stream(x) / self._sig_scales
        return np.cumsum((sigs @ self.l_star) ** 2)
 
 
# ---------------------------------------------------------------------------
# Numerically-stable sigmoid  σ(z) = 1 / (1 + e^{-z})
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
 