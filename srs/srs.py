"""Shock response spectrum (maximax absolute acceleration), Smallwood ramp-invariant filter.

Reference utility (Python). Personal, generic textbook algorithm; no NASA data or code.
srs(accel, fs, fn, Q) -> (pos, neg, maximax) arrays, one value per natural frequency.
Includes residual response: each oscillator rings on with zero input for 1.5 periods after the record.
"""
import numpy as np
from scipy.signal import lfilter


def smallwood_coeffs(fn, fs, Q):
    wn = 2 * np.pi * fn
    zeta = 1.0 / (2.0 * Q)
    wd = wn * np.sqrt(1.0 - zeta ** 2)
    dt = 1.0 / fs
    E = np.exp(-zeta * wn * dt)
    K = wd * dt
    C = E * np.cos(K)
    S = E * np.sin(K)
    Sp = S / K
    b = np.array([1.0 - Sp, 2.0 * (Sp - C), E ** 2 - Sp])
    a = np.array([1.0, -2.0 * C, E ** 2])
    return b, a


def srs(accel, fs, fn, Q=10.0):
    accel = np.asarray(accel, dtype=float)
    fn = np.atleast_1d(np.asarray(fn, dtype=float))
    pos = np.empty(fn.size)
    neg = np.empty(fn.size)
    for i, f in enumerate(fn):
        b, a = smallwood_coeffs(f, fs, Q)
        npad = int(np.ceil(1.5 * fs / f))   # residual: free decay peaks within the first period
        y = lfilter(b, a, np.concatenate([accel, np.zeros(npad)]))
        pos[i] = y.max()
        neg[i] = -y.min()
    return pos, neg, np.maximum(pos, neg)
