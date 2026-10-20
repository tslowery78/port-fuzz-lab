/* C port of PyLnD loads/rf_functions.py (recurrence formulas, piecewise-linear load).
 * Arrays: t[n], p[n] (sdof); p[n_modes*n_points] row-major (mdof). u/ud hold u0/ud0 in [0]
 * and are filled in place, like the Python version. */
#include <math.h>

void rf_coefficients(double dt, double k, double omn, double zeta, double c8[8]) {
    double beta = zeta * omn, h = dt, omd = omn * sqrt(1 - zeta * zeta);
    double s = sin(omd * h), co = cos(omd * h);
    double t1 = 1 / (k * omd * h), t2 = exp(-beta * h);
    double t3 = ((omd * omd - beta * beta) / (omn * omn) - beta * h) * s;
    double t4 = (2 * omd * beta / (omn * omn) + omd * h) * co;
    double t5 = 2 * beta * omd / (omn * omn);
    c8[0] = t1 * (t2 * (t3 - t4) + t5);
    double bt3 = ((omd * omd - beta * beta) / (omn * omn)) * s;
    double bt4 = (2 * omd * beta / (omn * omn)) * co;
    c8[1] = t1 * (t2 * (-bt3 + bt4) + omd * h - t5);
    c8[2] = t2 * (co + (beta / omd) * s);
    c8[3] = (1 / omd) * t2 * s;
    double apt4 = omd * co;
    c8[4] = t1 * (t2 * ((beta + omn * omn * h) * s + apt4) - omd);
    c8[5] = t1 * (-t2 * (beta * s + apt4) + omd);
    c8[6] = -(omn * omn / omd) * t2 * s;
    c8[7] = t2 * (co - (beta / omd) * s);
}

void rf_sdof(const double *t, const double *p, long n, double k, double omn, double zeta,
             double *u, double *ud) {
    double c[8] = {0}, last_dt = 0.0;
    for (long i = 0; i < n - 1; i++) {
        double dt = t[i + 1] - t[i];
        if (fabs(dt - last_dt) > 0.00001) rf_coefficients(dt, k, omn, zeta, c);
        u[i + 1]  = c[0] * p[i] + c[1] * p[i + 1] + c[2] * u[i] + c[3] * ud[i];
        ud[i + 1] = c[4] * p[i] + c[5] * p[i + 1] + c[6] * u[i] + c[7] * ud[i];
        last_dt = dt;
    }
}

void rf_mdof(const double *t, const double *p, long nm, long np, const double *k,
             const double *omn, const double *zeta, double *eta, double *etad) {
    double c[64][8] = {{0}}, last_dt = 0.0;   /* up to 64 modes in the fuzz */
    for (long i = 0; i < np - 1; i++) {
        double dt = t[i + 1] - t[i];
        if (fabs(dt - last_dt) > 0.00001)
            for (long j = 0; j < nm; j++) rf_coefficients(dt, k[j], omn[j], zeta[j], c[j]);
        last_dt = dt;
        for (long j = 0; j < nm; j++) {
            const double *q = p + j * np; double *e = eta + j * np, *ed = etad + j * np;
            e[i + 1]  = c[j][0] * q[i] + c[j][1] * q[i + 1] + c[j][2] * e[i] + c[j][3] * ed[i];
            ed[i + 1] = c[j][4] * q[i] + c[j][5] * q[i + 1] + c[j][6] * e[i] + c[j][7] * ed[i];
        }
    }
}
