import numpy as np


#EMPIRICAL MEAN REVERSION TIME :    

def find_important_extrema(series, C : float = 2.0):

    """ Returns list of (index, 'min'|'max') for all important extrema.
         A point is an important min if both surrounding endpoints are >= C*std above it.    
    """

    s       = np.array(series)
    std     = s.std()
    extrema = []
    n       = len(s)

    for m in range(1, n-1):
        #Check for important maximum along the serie array
        left_max  = s[:m].max() if m>0 else s[0]
        right_max = s[m:].max() if m<n-1 else s[-1]

        if (left_max - s[m] >= C*std) and (right_max-s[m] >= C*std):
            if s[m] == s[max(0,m-5):m+6].min():
                extrema.append((m,'min')) #local minnimum check
                continue
        
        left_min  = s[:m].min() if m>0 else s[0]
        right_min = s[m:].min() if m<n-1 else s[-1]
        if (s[m] - left_min >= C*std) and (s[m]-right_min>=C*std):
            if s[m] == s[max(0,m-5):m+6].max():
                extrema.append((m,'max'))
        
    return extrema

def compute_emrt(series, C:float = 2.0):

    """
        Empirical Mean Reversion Time : Avc time from a local extreme back to mean

    """
    
    s     = np.array(series)
    mean  = s.mean()
    extr  = find_important_extrema(s, C)

    if len(extr) < 2:
        return np.inf
    
    durations = []
    for idx, kind in extr :
        future = s[idx:]
        if kind == 'min':
            crossing = np.where(future>=mean)[0]
        else :
            crossing = np.where(future <=mean)[0]
        if len(crossing) > 0:
            durations.append(crossing[0])
    
    return float(np.mean(durations)) if durations else np.inf

def fit_hedge_ratio_emrt(s1, s2, grid_min=-3.0, grid_max=3.0, grid_step=0.01,C=2.0, max_var=None):

    """
    Finding B minimizing EMRT of (s1 - B*s2)"""

    s1 = np.array(s1)
    s2 = np.array(s2)
    b_values  = np.arange(grid_min, grid_max + grid_step, grid_step)
    best_b    = 1.0
    best_emrt = np.inf
    for b in b_values:
        spread = s1 - b * s2
        if max_var is not None and spread.var() > max_var:
            continue
        emrt = compute_emrt(spread, C)
        if emrt < best_emrt:
            best_emrt = emrt
            best_b    = b
    
    return best_b, best_emrt