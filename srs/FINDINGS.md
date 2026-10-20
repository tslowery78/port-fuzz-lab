# SRS port + differential fuzz (2026-10-20)
Utility: Smallwood ramp-invariant SRS, maximax absolute accel (srs.py). Port: srs.c (ctypes). Generic textbook algorithm, no NASA code/data.
No MATLAB/Octave on the Macs, so the port target is C (same idea as the article's C->Rust).

Run: python3 fuzz.py <seed> <trials>   (seed 1, 3000 trials = 11,948 cases, 23 s)
- Python vs C, valid inputs (Q>0.5): 0 mismatches; worst rel diff 1.3e-11.
- Smallwood vs exact first-order-hold state-space (cont2discrete foh): worst rel 4e-11 for fn/fs 0-0.5.
Mismatches found (all real, all in BOTH versions):
1. Q = 0.5 (critical damping): wd=0 -> 0/0. Python returns NaN, C returns -inf. Neither errors.
2. Q < 0.5: sqrt of negative -> same NaN / -inf. No input check.
3. fn >= fs/2: both return a finite, aliased number silently (970 cases).
4. NaN in input (from reading the code, not fuzzed): C max skips NaN, Python propagates it.
Engineering check (between_samples.py, 1 ms 100 g half-sine, Q=10, fs=10 kHz):
- Sampled peak misses the true peak by 0.1% (fs/fn=50), 2.0% (10), 5.4% (2.5).
- Fix options: validate Q>0.5 (or handle Q=0.5 via limit Sp->1), reject fn>=fs/2, upsample when fs/fn<10.
Second-model review fix (2026-10-20): residual response added (1.5 periods of zero input after the record, both
versions and the exact reference). [0,100,0] g at fs=1 kHz, fn=10 Hz, Q=10 went 1.016 g -> 5.848 g.
1.5 vs 20 periods: 0 shortfall in 300 random cases. Re-run: port worst 8e-11, vs exact worst 3e-10.
