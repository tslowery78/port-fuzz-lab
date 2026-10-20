/* C port of srs.py: Smallwood ramp-invariant SRS, maximax absolute acceleration.
 * Includes residual response: free decay after the record until its envelope is below the peaks found.
 * int srs(const double *accel, long n, double fs, const double *fn, long nf, double Q,
 *         double *pos, double *neg, double *maximax);  returns 0 ok, -1 bad args. */
#include <math.h>

static void coeffs(double fn, double fs, double Q, double b[3], double a[3]) {
    double wn = 2.0 * M_PI * fn;
    double zeta = 1.0 / (2.0 * Q);
    double wd = wn * sqrt(1.0 - zeta * zeta);
    double dt = 1.0 / fs;
    double E = exp(-zeta * wn * dt);
    double K = wd * dt;
    double C = E * cos(K);
    double S = E * sin(K);
    double Sp = S / K;
    b[0] = 1.0 - Sp; b[1] = 2.0 * (Sp - C); b[2] = E * E - Sp;
    a[0] = 1.0; a[1] = -2.0 * C; a[2] = E * E;
}

/* Same closed-form free-decay tail as residual_tail() in srs.py. */
static void residual_tail(double y0, double y1, double fn, double fs, double Q, double *pos, double *neg) {
    double zeta = 1.0 / (2.0 * Q), wn = 2.0 * M_PI * fn;
    double E = exp(-zeta * wn / fs), K = wn * sqrt(1.0 - zeta * zeta) / fs, s = sin(K);
    double lo = fmin(*pos, *neg) > 0 ? fmin(*pos, *neg) : fmax(*pos, *neg);
    if (!(isfinite(lo) && lo > 0 && E > 0 && E < 1 && fabs(s) > 1e-12)) return;
    double P = y0, R = (y1 / E - P * cos(K)) / s, M = hypot(P, R);
    if (M <= lo) return;
    double ex = ceil(log(M / lo) / -log(E));
    long extra = ex < 1e7 ? (long)ex : 10000000L;
    for (long k = 2; k < extra + 2; k++) {
        double y = pow(E, (double)k) * (P * cos(k * K) + R * sin(k * K));
        if (y > *pos) *pos = y;
        if (-y > *neg) *neg = -y;
    }
}

int srs(const double *accel, long n, double fs, const double *fn, long nf, double Q,
        double *pos, double *neg, double *maximax) {
    if (n < 1 || nf < 1) return -1;
    for (long i = 0; i < nf; i++) {
        double b[3], a[3];
        coeffs(fn[i], fs, Q, b, a);
        /* direct form II transposed, zero initial state (matches scipy lfilter) */
        double z1 = 0.0, z2 = 0.0, ymax = -INFINITY, ymin = INFINITY;
        long npad = (long)ceil(1.5 * fs / fn[i]);
        if (npad < 2) npad = 2;
        double yprev = 0.0, ylast = 0.0;
        for (long k = 0; k < n + npad; k++) {
            double x = k < n ? accel[k] : 0.0;
            double y = b[0] * x + z1;
            z1 = b[1] * x - a[1] * y + z2;
            z2 = b[2] * x - a[2] * y;
            if (y > ymax) ymax = y;
            if (y < ymin) ymin = y;
            yprev = ylast; ylast = y;
        }
        pos[i] = ymax; neg[i] = -ymin;
        residual_tail(yprev, ylast, fn[i], fs, Q, &pos[i], &neg[i]);
        maximax[i] = pos[i] > neg[i] ? pos[i] : neg[i];
    }
    return 0;
}
