import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math 
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

import core.ou_fitting as ou 
from strategies.pairs_trading import OUpairTrader as pt
from core.spread import construct_spread
from data.loader import fetch_pair_prices


class SignatureComputation :

    def __init__(self, config):

        self.config = config

        self.spread_serie   = None
        self.spread_price   = None
        self.path           = None

        self.seg_signatures     = []
        self.l2_path_signature  = []

        self.sig_l              = None
        self.N                  = None

    def segment_signature(self, delta):

        delta = np.asarray(delta, dtype=float)
        sig   = {0:np.array(1.0)}
        term  = np.array(1.0)
        for k in range(1, self.N +1):
            term = np.tensordot(term, delta, axes = 0) / math.factorial(k) 
            sig[k] =  term
        return sig 

    def tensor_mult(a, b, N):
        out = {}
        for m in range(N+1):
            acc = None 
            for k in range(m+1):
                prod = np.tensordot(a[k], b[m-k], axes = 0)
                acc  = prod if acc is None else acc + prod
            out[m] = acc
        return out 
    
    def path_signature(path,N):
        path = np.asarray(path, dtype=float)
        d = path.shape[1]
        sig = {0:np.array(1.0)}
        for lvl in range(1,N+1):
            sig[lvl]= np.zeros((d,)*lvl)
        for i in range(len(path) - 1):
            seg = SignatureComputation.segment_signature(path[i+1] - path[i])
            sig = SignatureComputation.tensor_mult(sig, seg, N)
        return sig 
    
    def time_augment(x, T = 1.0):
        x = np.asarray(x, dtype=float)
        t = np.linspace(0.0, T, len(x))
        return np.column_stack([t,x])
    
    def flatten(self,sig):
        parts = [np.atleast_1d(sig[k]).ravel() for k in range(self.N + 1)]
        return np.concatenate(parts)
    
    def sig_length(d,self):
        return sum(d ** k for k in range(self.N + 1))
    
    def main(self):

        










    


    

