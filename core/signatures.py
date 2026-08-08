import math 
import numpy as np


#Building Blocks

def segment_signature(self, delta):

    delta = np.asarray(delta, dtype=float)
    sig   = {0:np.array(1.0)}
    term  = np.array(1.0)
    for k in range(1, self.N +1):
        term = np.tensordot(term, delta, axes = 0) / math.factorial(k) 
        sig[k] =  term
    return sig 

def tensor_mult(a: dict, b: dict, N: int) -> dict:
    """
    Concatenation product Sig(A) ⊗ Sig(B), truncated at level N (fact F2).
        out[m] = Σ_{k=0..m}  a[k] ⊗ b[m-k]
    """
    out = {}
    for m in range(N + 1):
        acc = None
        for k in range(m + 1):
            prod = np.tensordot(a[k], b[m - k], axes=0)
            acc = prod if acc is None else acc + prod
        out[m] = acc
    return out

def path_signature(path: np.ndarray, N: int) -> dict:
    """
    Truncated signature of a polyline. `path` has shape (n_points, d).
    We start from Sig(empty) = 1, walk the segments, and Chen-multiply.
    """
    path = np.asarray(path, dtype=float)
    d = path.shape[1]
    sig = {0: np.array(1.0)}
    for lvl in range(1, N + 1):
        sig[lvl] = np.zeros((d,) * lvl)
    for i in range(len(path) - 1):
        seg = segment_signature(path[i + 1] - path[i], N)
        sig = tensor_mult(sig, seg, N)
    return sig
 

 #Utilities 

def time_augment(x, T: float = 1.0) -> np.ndarray:
    """
    Turn a 1D series x[0..n] into the 2D augmented path (t, x_t).
    Time is coordinate 0 so the signature loses no info (kills the
    time-parametrization invariance).
    """
    x = np.asarray(x, dtype=float)
    t = np.linspace(0.0, T, len(x))
    return np.column_stack([t, x])

def flatten(sig: dict, N: int) -> np.ndarray:
    """Flatten the level-dict into one plain vector so ⟨l, Sig⟩ is a dot product."""
    parts = [np.atleast_1d(sig[k]).ravel() for k in range(N + 1)]
    return np.concatenate(parts)

def sig_length(d: int, N: int) -> int:
    """Length of the flattened truncated signature = 1 + d + d² + ... + d^N."""
    return sum(d ** k for k in range(N + 1))

def path_signature_stream(path: np.ndarray, N: int) -> list:
    """
    Signatures of ALL prefixes of the path:
        returns [Sig(X[0:1]), Sig(X[0:2]), ..., Sig(X[0:n])]
    (each entry is a flattened vector).
    This is exactly what the signature optimal stopping loop needs:
    X̂^{≤N}_{0, t_i} for every i.
    """
    path = np.asarray(path, dtype=float)
    d = path.shape[1]
    sig = {0: np.array(1.0)}
    for lvl in range(1, N + 1):
        sig[lvl] = np.zeros((d,) * lvl)
 
    stream = [flatten(sig, N)]                # Sig over [0, t_0] = trivial
    for i in range(len(path) - 1):
        seg = segment_signature(path[i + 1] - path[i], N)
        sig = tensor_mult(sig, seg, N)
        stream.append(flatten(sig, N))
    return stream