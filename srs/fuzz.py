"""Differential fuzz: srs.py vs C port (libsrs.dylib), plus an exact first-order-hold
state-space reference (scipy cont2discrete 'foh' = exact for piecewise-linear input)."""
import ctypes, json, sys, numpy as np
from pathlib import Path
from scipy.signal import cont2discrete, dlsim
sys.path.insert(0, str(Path(__file__).parent))
import srs as pysrs

lib = ctypes.CDLL(str(Path(__file__).parent / "libsrs.dylib"))
P = ctypes.POINTER(ctypes.c_double)
lib.srs.argtypes = [P, ctypes.c_long, ctypes.c_double, P, ctypes.c_long, ctypes.c_double, P, P, P]

def csrs(x, fs, fn, Q):
    x = np.ascontiguousarray(x, float); fn = np.ascontiguousarray(np.atleast_1d(fn), float)
    out = [np.empty(fn.size) for _ in range(3)]
    rc = lib.srs(x.ctypes.data_as(P), x.size, fs, fn.ctypes.data_as(P), fn.size, Q,
                 *[o.ctypes.data_as(P) for o in out])
    return rc, out

def exact(x, fs, f, Q):
    wn, z = 2*np.pi*f, 1/(2*Q)
    A = [[0, 1], [-wn**2, -2*z*wn]]; B = [[0], [-1]]; C = [[-wn**2, -2*z*wn]]; D = [[0]]
    sysd = cont2discrete(tuple(np.array(m, float) for m in (A, B, C, D)), 1/fs, method="foh")
    x = np.concatenate([x, np.zeros(int(np.ceil(1.5 * fs / f)))])   # same residual window as srs()
    _, y, _ = dlsim(sysd, x)
    y = y.ravel(); return max(y.max(), -y.min())

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
def signal(kind, n, fs):
    t = np.arange(n)/fs
    if kind == "noise": return rng.normal(0, rng.uniform(0.1, 100), n)
    if kind == "halfsine":
        T = rng.uniform(5, 50)*1/fs; x = np.where(t < T, np.sin(np.pi*t/T), 0.0); return rng.uniform(1, 500)*x
    if kind == "decay":
        f = rng.uniform(1, fs/3); return rng.uniform(1, 1e3)*np.exp(-rng.uniform(1, 50)*t)*np.sin(2*np.pi*f*t)
    if kind == "spike": x = np.zeros(n); x[rng.integers(0, n)] = rng.uniform(-1e4, 1e4); return x
    if kind == "const": return np.full(n, rng.uniform(-10, 10))
    if kind == "huge": return rng.normal(0, 1e150, n)
    if kind == "tiny": return rng.normal(0, 1e-300, n)

rows = []; worst_port = 0.0
for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3000):
    kind = rng.choice(["noise", "halfsine", "decay", "spike", "const", "huge", "tiny"])
    fs = float(rng.choice([1e3, 5e3, 1e4, 1e5]) * rng.uniform(0.5, 2))
    n = int(rng.choice([1, 2, 3, 50, 500, 5000]))
    Q = float(rng.choice([rng.uniform(0.51, 100), 10.0, 0.5, 0.3]))   # include Q<=0.5 edge cases
    fn = np.sort(rng.uniform(0.5, 0.6*fs, int(rng.integers(1, 8))))      # some above Nyquist on purpose
    x = signal(kind, n, fs)
    with np.errstate(all="ignore"):
        p = pysrs.srs(x, fs, fn, Q)
    rc, c = csrs(x, fs, fn, Q)
    for j, f in enumerate(fn):
        pv, cv = p[2][j], c[2][j]
        both_nan = np.isnan(pv) and np.isnan(cv)
        if both_nan: d = 0.0
        elif np.isnan(pv) != np.isnan(cv) or np.isinf(pv) != np.isinf(cv): d = np.inf
        else: d = abs(pv - cv) / max(abs(pv), 1e-300) if pv != cv else 0.0
        worst_port = max(worst_port, d if np.isfinite(d) else worst_port)
        r = dict(trial=trial, kind=str(kind), n=n, fs=fs, fn=float(f), Q=Q, fn_over_fs=float(f/fs),
                 py=float(pv), c=float(cv), port_rel=float(d))
        if Q > 0.5 and f < fs/2 and n > 1 and np.isfinite(pv) and kind not in ("huge", "tiny"):
            ex = exact(x, fs, f, Q); r["exact"] = float(ex)
            r["exact_rel"] = float(abs(pv - ex)/ex) if ex > 0 else 0.0
        rows.append(r)

out = Path(__file__).parent / "fuzz_results.json"
out.write_text(json.dumps(rows))
bad = [r for r in rows if r["port_rel"] > 1e-9]
print(f"cases {len(rows)}  port mismatches >1e-9: {len(bad)}  worst finite port rel: {worst_port:.2e}")
from collections import Counter
print("mismatch causes:", Counter((r["kind"], "Q<=.5" if r["Q"] <= .5 else "Q ok", "fn>=Nyq" if r["fn_over_fs"] >= .5 else "fn<Nyq",
                                  "py=nan" if np.isnan(r["py"]) else f"py={r['py']:.3g}", f"c={r['c']:.3g}") for r in bad).most_common(12))
ex = [r for r in rows if "exact_rel" in r]
for lo, hi in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5)]:
    s = [r["exact_rel"] for r in ex if lo <= r["fn_over_fs"] < hi]
    if s: print(f"fn/fs {lo:.2f}-{hi:.2f}: n={len(s):4d}  Smallwood vs exact: median {np.median(s):.2e}  max {max(s):.2e}")
