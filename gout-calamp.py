from scipy.special import gamma, gammainc
from scipy.misc import comb
from math import sqrt


def sign(x):
    if (x < 0):
        return -1
    else:
        return 1


def integral_p(N, mu, sigma, a, b):
    result = 0
    for i in range(N + 1):
        gam = gamma(i / 2. + 0.5)
        gam_ib = gammainc(i / 2. + 0.5, (b - mu)**2 / (2 * sigma))
        gam_ia = gammainc(i / 2. + 0.5, (a - mu)**2 / (2 * sigma))
        if (i % 2 == 0):
            incr = 0.5 * gam * (sign(b - mu) * gam_ib - sign(a - mu) * gam_ia)
        else:

            incr = 0.5 * gam * (gam_ib - gam_ia)
        result = result + \
            comb(N, i) * mu**(N - i) * (sqrt(2 * sigma))**(i + 1) * incr
    return result


def posterior_m_and_v_on_d(P, wd, MD, VD):
    eps = 1E-8
    binf = (2 - wd) / 2.
    bsup = (2 + wd) / 2.

    int0 = integral_p(P, MD, VD, binf, bsup)
    int1 = integral_p(P + 1, MD, VD, binf, bsup)
    int2 = integral_p(P + 2, MD, VD, binf, bsup)

    md = int1 / max(int0, eps)
    vd = max(int2 / max(int0, eps) - md**2, eps)

    print(int1, int2, int0)
    return md, vd


(m, v) = posterior_m_and_v_on_d(2, 1, 1.5100000, 1)
print(m, v)
