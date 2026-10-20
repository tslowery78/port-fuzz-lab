from rf_harness import *
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rows = []
for trial in range(N):
    tk = rng.choice(["uniform", "uniform", "variable", "jitter_small", "tiny_dt", "repeat_time"])
    n = int(rng.choice([2, 3, 10, 200, 2000]))
    w = 2*np.pi*float(rng.uniform(0.1, 200))
    dt = float(rng.uniform(0.02, 0.2)) / (w/(2*np.pi))           # 5..50 points per cycle
    if tk == "uniform": t = np.arange(n) * dt
    elif tk == "variable": t = np.concatenate([[0], np.cumsum(dt * rng.uniform(0.3, 1.7, n-1))])
    elif tk == "jitter_small":                                        # dt changes by less than 1e-5 s
        t = np.concatenate([[0], np.cumsum(dt + rng.uniform(-4e-6, 4e-6, n-1))])
    elif tk == "tiny_dt": dt = float(rng.uniform(1e-7, 9e-6)); t = np.arange(n) * dt
    else: t = np.arange(n) * dt; t[n//2:] = t[n//2:] - dt if n > 2 else t[n//2:]   # duplicate time stamp
    z = float(rng.choice([rng.uniform(0, 0.2), rng.uniform(0.2, 0.99), 0.0, 1.0, 1.5]))
    k = float(rng.choice([w**2, w**2 * rng.uniform(0.01, 100)]))     # unit mass modal, or physical stiffness
    p = rng.normal(0, rng.uniform(0.1, 1e3), n); u0, v0 = rng.normal(0, 1, 2) * rng.choice([0, 1])
    r = dict(trial=trial, tkind=str(tk), n=n, zeta=z, dt=float(t[1]-t[0]))
    try:
        with np.errstate(all="ignore"): pu, pv = py_sdof(t, p, k, w, z, u0, v0); r["py"] = "ok"
    except Exception as e: r["py"] = f"{type(e).__name__}: {str(e)[:60]}"; pu = None
    with np.errstate(all="ignore"): cu, cv = c_sdof(t, p, k, w, z, u0, v0)
    r["c_finite"] = bool(np.all(np.isfinite(cu)))
    r["port_rel"] = rel(cu, pu) if pu is not None else np.inf
    if z < 1 and np.all(np.diff(t) > 0):
        eu, ev = exact_sdof(t, p, k, w, z, u0, v0)
        r["py_vs_exact"] = rel(pu, eu) if pu is not None else np.inf
        r["c_vs_exact"] = rel(cu, eu)
    rows.append(r)

# MDOF vs SDOF consistency (same Python file): each mode of rf_mdof should equal rf_sdof
mdof_bad = 0
for trial in range(200):
    nm, npnt = int(rng.integers(1, 6)), int(rng.choice([2, 50, 500]))
    w = 2*np.pi*rng.uniform(1, 50, nm); z = rng.uniform(0, 0.5, nm); k = w**2
    t = np.arange(npnt) * 1e-3; p = rng.normal(0, 10, (nm, npnt))
    eta = np.zeros((nm, npnt)); etad = np.zeros((nm, npnt))
    e1, _ = rfp.rf_mdof(t, p, k, w, z, eta.copy(), etad.copy())
    ec = np.zeros(nm*npnt); edc = np.zeros(nm*npnt)
    lib.rf_mdof(ptr(t), ptr(np.ascontiguousarray(p)), nm, npnt, ptr(k), ptr(w), ptr(z), ptr(ec), ptr(edc))
    for j in range(nm):
        s, _ = py_sdof(t, p[j], k[j], w[j], z[j], 0.0, 0.0)
        if rel(e1[j], s) > 1e-9 or rel(ec.reshape(nm, npnt)[j], s) > 1e-9: mdof_bad += 1

(HERE / "fuzz_rf_results.json").write_text(json.dumps(rows, default=float))
print(f"SDOF cases {len(rows)}; MDOF modes inconsistent with SDOF: {mdof_bad}")
good = [r for r in rows if r["py"] == "ok" and np.isfinite(r["port_rel"])]
print(f"Python vs C (both ran): worst rel {max(r['port_rel'] for r in good):.1e}")
print("Python crashes:", Counter((r["tkind"], r["py"]) for r in rows if r["py"] != "ok").most_common(6))
print("Non-finite output by zeta:", Counter(r["zeta"] for r in rows if not r["c_finite"]).most_common(4))
for tk in ["uniform", "variable", "jitter_small", "tiny_dt"]:
    s = [r["py_vs_exact"] for r in rows if r["tkind"] == tk and "py_vs_exact" in r and np.isfinite(r["py_vs_exact"])]
    c = [r["c_vs_exact"] for r in rows if r["tkind"] == tk and "c_vs_exact" in r]
    if c: print(f"{tk:13s} vs exact: Python median {np.median(s) if s else float('nan'):.1e} max {max(s) if s else float('nan'):.1e} (n={len(s)}) | C median {np.median(c):.1e} max {max(c):.1e}")
