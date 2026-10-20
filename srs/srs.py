"""Shock response spectrum (maximax absolute acceleration), Smallwood ramp-invariant filter.

Reference utility (Python). Personal, generic textbook algorithm; no NASA data or code.
srs(accel, fs, fn, Q) -> (pos, neg, maximax) arrays, one value per natural frequency.
Includes residual response: after the record each oscillator rings freely (zero input) until its decay
envelope drops below the peaks already found, so no later sampled peak can be missed.
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


def residual_tail(y0, y1, fn, fs, Q, pos, neg, cap=10_000_000):
    """Free decay after the padded record: y_k = E^k (P cos kK + R sin kK), k=0 at y0.
    Extend until the envelope E^k * hypot(P, R) is below min(pos, neg)."""
    zeta = 1.0 / (2.0 * Q)
    wn = 2 * np.pi * fn
    E = np.exp(-zeta * wn / fs)
    K = wn * np.sqrt(1.0 - zeta ** 2) / fs
    s = np.sin(K)
    lo = min(pos, neg) if min(pos, neg) > 0 else max(pos, neg)
    if not (np.isfinite(lo) and lo > 0 and 0 < E < 1 and abs(s) > 1e-12):
        return pos, neg
    P = y0
    R = (y1 / E - P * np.cos(K)) / s
    M = np.hypot(P, R)
    if M <= lo:
        return pos, neg
    extra = min(int(np.ceil(np.log(M / lo) / -np.log(E))), cap)
    k = np.arange(2, extra + 2, dtype=float)
    yt = E ** k * (P * np.cos(k * K) + R * np.sin(k * K))
    return max(pos, yt.max()), max(neg, -yt.min())


def srs(accel, fs, fn, Q=10.0):
    accel = np.asarray(accel, dtype=float)
    fn = np.atleast_1d(np.asarray(fn, dtype=float))
    pos = np.empty(fn.size)
    neg = np.empty(fn.size)
    for i, f in enumerate(fn):
        b, a = smallwood_coeffs(f, fs, Q)
        npad = max(2, int(np.ceil(1.5 * fs / f)))
        y = lfilter(b, a, np.concatenate([accel, np.zeros(npad)]))
        pos[i], neg[i] = y.max(), -y.min()
        pos[i], neg[i] = residual_tail(y[-2], y[-1], f, fs, Q, pos[i], neg[i])
    return pos, neg, np.maximum(pos, neg)
