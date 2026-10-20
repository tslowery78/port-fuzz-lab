"""How much does sampled maximax miss the true (between-sample) peak? Half-sine 1 ms, 100 g, Q=10."""
import sys, numpy as np; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import srs as pysrs
fs, Q, T = 10_000.0, 10.0, 1e-3
t = np.arange(int(0.05*fs))/fs; x = np.where(t < T, 100*np.sin(np.pi*t/T), 0.0)
up = 64; tf = np.arange(len(t)*up)/(fs*up); xf = np.interp(tf, t, x)   # same piecewise-linear input, finer grid
for ratio in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49]:
    f = ratio*fs
    coarse = pysrs.srs(x, fs, f, Q)[2][0]; fine = pysrs.srs(xf, fs*up, f, Q)[2][0]
    print(f"fn/fs={ratio:4.2f} (fs/fn={1/ratio:5.1f}): sampled {coarse:8.2f} g  true {fine:8.2f} g  under by {100*(1-coarse/fine):5.1f}%")
