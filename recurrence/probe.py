import sys, numpy as np; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import rf_functions as rfp
from rf_harness import py_sdof, c_sdof, exact_sdof, rel, lib, ptr
# F: MDOF vs SDOF
rng = np.random.default_rng(5)
for trial in range(300):
    nm, npnt = int(rng.integers(1, 6)), int(rng.choice([2, 50, 500]))
    w = 2*np.pi*rng.uniform(1, 50, nm); z = rng.uniform(0, 0.5, nm); k = w**2
    t = np.arange(npnt) * 1e-3; p = rng.normal(0, 10, (nm, npnt))
    e1, _ = rfp.rf_mdof(t, p, k, w, z, np.zeros((nm, npnt)), np.zeros((nm, npnt)))
    errs = [rel(e1[j], py_sdof(t, p[j], k[j], w[j], z[j], 0., 0.)[0]) for j in range(nm)]
    if max(errs) > 1e-12: print("MDOF diff", nm, npnt, ["%.1e" % e for e in errs]); break
# C: accuracy vs omega*h, uniform steps, zeta 0.05
w, z = 2*np.pi*10, 0.05
for wh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    h = wh / w; t = np.arange(400) * h; p = np.sin(2*np.pi*3*t) + 1
    try: pu = py_sdof(t, p, w**2, w, z, 0., 0.)[0]; pe = rel(pu, exact_sdof(t, p, w**2, w, z, 0., 0.)[0])
    except Exception as e: pe = type(e).__name__
    ce = rel(c_sdof(t, p, w**2, w, z, 0., 0.)[0], exact_sdof(t, p, w**2, w, z, 0., 0.)[0])
    print(f"omega*h={wh:.0e} (h={h:.1e} s, {2*np.pi/wh:,.0f} pts/cycle): Python err {pe if isinstance(pe,str) else f'{pe:.1e}'}  C err {ce:.1e}")
# C2: coefficient accuracy alone (bypasses the 1e-5 s recompute check), vs exact one-step map
from scipy.linalg import expm
for wh in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    h = wh / w; m = 1.0; A = np.array([[0, 1], [-w**2, -2*z*w]])
    M = np.zeros((4, 4)); M[:2, :2] = A; M[:2, 2] = [0, 1]; M[2, 3] = 1; E = expm(M*h)
    # exact: x1 = Phi x0 + G0 p0 + G1 (p1-p0)/h  ->  coefficient on p0 = G0 - G1/h, on p1 = G1/h
    a_ex, b_ex = E[0, 2] - E[0, 3]/h, E[0, 3]/h
    a, b, *_ = rfp.rf_coefficients(h, w**2, w, z)
    print(f"omega*h={wh:.0e}: rel err a {abs(a-a_ex)/abs(a_ex):.1e}  b {abs(b-b_ex)/abs(b_ex):.1e}")
# F2: C MDOF vs Python MDOF
rng = np.random.default_rng(7); worst = 0
for trial in range(300):
    nm, npnt = int(rng.integers(1, 6)), int(rng.choice([2, 50, 500]))
    w2 = 2*np.pi*rng.uniform(1, 50, nm); z2 = rng.uniform(0, 0.5, nm); k = w2**2
    t = np.arange(npnt) * 1e-3; p = rng.normal(0, 10, (nm, npnt))
    e1, _ = rfp.rf_mdof(t, p, k, w2, z2, np.zeros((nm, npnt)), np.zeros((nm, npnt)))
    ec = np.zeros(nm*npnt); edc = np.zeros(nm*npnt)
    lib.rf_mdof(ptr(t), ptr(np.ascontiguousarray(p)), nm, npnt, ptr(k), ptr(w2), ptr(z2), ptr(ec), ptr(edc))
    e = rel(ec.reshape(nm, npnt), e1); worst = max(worst, e)
    if e > 1e-12: print("C MDOF diff", nm, npnt, f"{e:.1e}", "dt check:", abs(1e-3 - 0) > 1e-5); break
print("C vs Python MDOF worst", f"{worst:.1e}")
