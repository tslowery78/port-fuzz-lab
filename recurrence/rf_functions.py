#!/ots/sw/osstoolkit/15.1/sles11-x86_64/bin/python3.5
import math
import numpy as np


def rf_coefficients(dt, k, omn, zeta):
    """Function to calculate the Coefficients for Recurrence Formulas

        dt - delta time
        k - eigenvalue or stiffness
        omn - natural frequency [rad/s]
        zeta - % damping
    """

    beta = zeta * omn
    h = dt
    omd = omn * math.sqrt(1 - zeta**2)

    t1 = (1/(k * omd * h))
    t2 = math.exp(-beta * h)
    t3 = ((omd**2 - beta**2)/omn**2 - beta * h) * math.sin(omd * h)
    t4 = (2 * omd * beta / omn**2 + omd * h) * math.cos(omd * h)
    t5 = 2 * beta * omd / omn**2
    a = t1 * (t2 * (t3 - t4) + t5)

    bt3 = ((omd**2 - beta**2)/omn**2) * math.sin(omd * h)
    bt4 = (2 * omd * beta / omn**2) * math.cos(omd * h)
    bt5 = omd * h
    b = t1 * (t2 * (-bt3 + bt4) + bt5 - t5)

    ct2 = math.cos(omd * h)
    ct3 = (beta / omd) * math.sin(omd * h)
    c = t2 * (ct2 + ct3)

    d = (1 / omd) * t2 * math.sin(omd * h)

    apt3 = (beta + omn**2 * h) * math.sin(omd * h)
    apt4 = omd * math.cos(omd * h)
    ap = t1 * (t2 * (apt3 + apt4) - omd)

    bpt3 = beta * math.sin(omd * h)
    bp = t1 * (-t2 * (bpt3 + apt4) + omd)

    cp = - (omn**2 / omd) * t2 * math.sin(omd * h)

    dp = t2 * (math.cos(omd * h) - (beta / omd) * math.sin(omd * h))

    return [a, b, c, d, ap, bp, cp, dp]


def rf_sdof(t, p, k, omn, zeta, u0, ud0):
    """Function that integrates a SDOF EOM using Recurrence Formulas.

        t - time
        p - load vector
        k - eigenvalue or stiffness
        omn - natural frequencies [rad/s]
        zeta - % damping
        u0 - initial displacement
        ud0 - initial velocity
    """

    # Integrate the EOM to find the response displacement and velocity.
    n = t.size
    u = u0
    ud = ud0
    last_dt = 0.0
    for i in range(0, n - 1):
        delta_t = t[i + 1] - t[i]
        if abs(delta_t - last_dt) > 0.00001:
            [a, b, c, d, ap, bp, cp, dp] = rf_coefficients(delta_t, k, omn, zeta)
        u[i + 1] = a * p[i] + b * p[i + 1] + c * u[i] + d * ud[i]
        ud[i + 1] = ap * p[i] + bp * p[i + 1] + cp * u[i] + dp * ud[i]
        last_dt = delta_t

    return [u, ud]


def rf_mdof(t, p, k, omn, zeta, eta0, etad0):
    """Function that integrates a MDOF EOM using Recurrence Formulas.

        t - time
        p - load vector
        k - eigenvalue or stiffness
        omn - natural frequencies [rad/s]
        zeta - % damping
        eta - initial modal displacement
        eta0 - initial modal velocity
    """

    # Determine the size of the problem and initialize modal displacement and velocity.
    n_modes = p.shape[0]
    n_points = p.shape[1]
    eta = eta0
    etad = etad0

    # Integrate the MDOF EOM using Recurrence Formulas.
    last_dt = 0.0
    rfc = np.zeros([n_modes, 8])
    for i in range(0, n_points - 1):

        # Recalculate first two terms if the time step changes with appropriate Recurrence Coefficients.
        delta_t = t[i + 1] - t[i]
        if abs(delta_t - last_dt) > 0.00001:
            for j in range(0, n_modes):
                rfc[j, :] = rf_coefficients(delta_t, k[j], omn[j], zeta[j])
            a = rfc[:, 0]
            b = rfc[:, 1]
            c = rfc[:, 2]
            d = rfc[:, 3]
            ap = rfc[:, 4]
            bp = rfc[:, 5]
            cp = rfc[:, 6]
            dp = rfc[:, 7]
        last_dt = delta_t

        # Determine the response for this time step.
        t1 = np.einsum('i,i->i', a, p[:, i])
        t2 = np.einsum('i,i->i', b, p[:, i + 1])
        td1 = np.einsum('i,i->i', ap, p[:, i])
        td2 = np.einsum('i,i->i', bp, p[:, i + 1])
        t3 = np.einsum('i,i->i', c, eta[:, i])
        t4 = np.einsum('i,i->i', d, etad[:, i])
        td3 = np.einsum('i,i->i', cp, eta[:, i])
        td4 = np.einsum('i,i->i', dp, etad[:, i])
        eta[:, i + 1] = t1 + t2 + t3 + t4
        etad[:, i + 1] = td1 + td2 + td3 + td4

    return eta, etad
