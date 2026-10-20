"""Differential fuzz: PyLnD rf_functions.py vs C port (librf.dylib) vs exact state-space
(matrix exponential with first-order-hold load, exact for piecewise-linear p)."""
import ctypes, sys, json, numpy as np
from pathlib import Path
from collections import Counter
from scipy.linalg import expm
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import rf_functions as rfp

lib = ctypes.CDLL(str(HERE / "librf.dylib")); P = ctypes.POINTER(ctypes.c_double); d = ctypes.c_double; L = ctypes.c_long
lib.rf_sdof.argtypes = [P, P, L, d, d, d, P, P]
lib.rf_mdof.argtypes = [P, P, L, L, P, P, P, P, P]
ptr = lambda a: a.ctypes.data_as(P)

def c_sdof(t, p, k, w, z, u0, v0):
    u = np.zeros(t.size); v = np.zeros(t.size); u[0], v[0] = u0, v0
    lib.rf_sdof(ptr(t), ptr(p), t.size, k, w, z, ptr(u), ptr(v)); return u, v

def py_sdof(t, p, k, w, z, u0, v0):
    u = np.zeros(t.size); v = np.zeros(t.size); u[0], v[0] = u0, v0
    return rfp.rf_sdof(t, p, k, w, z, u, v)

def exact_sdof(t, p, k, w, z, u0, v0):
    m = k / w**2; A = np.array([[0, 1], [-w**2, -2*z*w]]); B = np.array([0, 1/m])
    x = np.array([u0, v0], float); out = [x.copy()]
    for i in range(t.size - 1):
        h = t[i+1] - t[i]
        # augmented: state, load p, load slope s; p(t)=p_i + s*tau
        M = np.zeros((4, 4)); M[:2, :2] = A; M[:2, 2] = B; M[2, 3] = 1
        E = expm(M * h); s = (p[i+1] - p[i]) / h
        x = E[:2, :2] @ x + E[:2, 2] * p[i] + E[:2, 3] * s; out.append(x.copy())
    o = np.array(out); return o[:, 0], o[:, 1]

def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape: return np.inf
    if np.array_equal(np.isnan(a), np.isnan(b)) and np.all(np.isnan(a) | (a == b)): return 0.0
    if np.any(np.isnan(a) != np.isnan(b)) or np.any(np.isinf(a) != np.isinf(b)): return np.inf
    sc = max(np.max(np.abs(b)), 1e-300); return float(np.max(np.abs(a - b)) / sc)

