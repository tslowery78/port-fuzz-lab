# Port + Fuzz Lab

Port a trusted engineering routine to a second language, feed both the same random inputs,
and chase every mismatch. Idea from Runtime Verification's C-to-Rust kernel work.

- `recurrence/`: PyLnD `rf_functions.py` (recurrence-formula SDOF/MDOF integrator) ported to C.
- `srs/`: Smallwood shock response spectrum, Python and C.

Each folder has a FINDINGS.md. Build the C side, then run the fuzz:

```
cd recurrence && clang -O2 -shared -fPIC -o librf.dylib rf.c && python3 fuzz_rf.py 1 2000
cd srs && clang -O2 -shared -fPIC -o libsrs.dylib srs.c && python3 fuzz.py 1 3000
```
(Linux: use `-o librf.so` and change the name in the harness.) Needs numpy and scipy.

Page: https://tslowery78.github.io/port-fuzz-lab/
