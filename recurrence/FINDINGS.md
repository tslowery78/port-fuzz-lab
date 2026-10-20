# PyLnD rf_functions.py -> C port + differential fuzz (2026-10-20)
Source: github.com/tslowery78/PyLnD loads/rf_functions.py (public, 2019). Port: rf.c (ctypes). Third check: exact
state-space step (expm, linear load between points) in rf_harness.py. Nothing pushed to the repo.
Run: python3 fuzz_rf.py <seed> <cases>   (seed 1, 2000 cases, 5 s); probe.py = accuracy vs omega*h.

Port itself: Python vs C agree to 2e-13 on normal inputs (797 cases); MDOF == SDOF per mode (1e-12).
The only Python/C gaps (up to 4e-3) came from bug 3 amplifying 1-ulp differences (omn**2 vs omn*omn).

Bugs (in the original Python):
1. First dt < 1e-5 s (sample rate > 100 kHz): coefficients never computed. Python: UnboundLocalError.
   C port: silently all-zero response. (Pyro-shock data is often 100 kHz-1 MHz.)
2. Recompute check is absolute (|dt - last_dt| > 1e-5 s): a dt change under 10 us keeps stale coefficients.
   Jittery timestamps: error up to 25-200% vs exact. Fix in rf_functions_fixed.py (relative 1e-9*dt): 1e-14.
3. Coefficient formula cancels for small omega*h: coefficient error 3e-8 at 6,300 pts/cycle, 4e-5 at 63,000,
   40x at 6.3M. Hits low modes with fine dt (0.1 Hz mode, dt 1e-4 s). Fix: Taylor series or expm for small omega*h.
4. zeta >= 1 crashes (ZeroDivisionError at 1, ValueError above); docstring says "% damping" but it is a fraction.
5. omn = 0 (rigid-body mode) crashes with ZeroDivisionError.
6. Repeated time stamp (dt = 0): inf/NaN, no error.
7. Interface: u0/ud0 and eta0/etad0 must be full-length arrays and are overwritten in place.
Second-model review fix (2026-10-20): C rf_mdof had a fixed 64-mode coefficient array (stack overflow past 64);
now heap-allocated for any mode count. 200-mode check vs Python: rel 9e-11.
