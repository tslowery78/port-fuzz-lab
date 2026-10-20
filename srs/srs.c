/* C port of srs.py: Smallwood ramp-invariant SRS, maximax absolute acceleration.
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

int srs(const double *accel, long n, double fs, const double *fn, long nf, double Q,
        double *pos, double *neg, double *maximax) {
    if (n < 1 || nf < 1) return -1;
    for (long i = 0; i < nf; i++) {
        double b[3], a[3];
        coeffs(fn[i], fs, Q, b, a);
        /* direct form II transposed, zero initial state (matches scipy lfilter) */
        double z1 = 0.0, z2 = 0.0, ymax = -INFINITY, ymin = INFINITY;
        for (long k = 0; k < n; k++) {
            double x = accel[k];
            double y = b[0] * x + z1;
            z1 = b[1] * x - a[1] * y + z2;
            z2 = b[2] * x - a[2] * y;
            if (y > ymax) ymax = y;
            if (y < ymin) ymin = y;
        }
        pos[i] = ymax; neg[i] = -ymin;
        maximax[i] = pos[i] > neg[i] ? pos[i] : neg[i];
    }
    return 0;
}
