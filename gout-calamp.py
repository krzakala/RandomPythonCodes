from scipy.special import *
from math import sqrt


def sign(x):
    if (x < 0):
        return -1
    else:
        return 1


def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n / k * nchoosek(n - 1, k - 1)
    return round(r)


def integral_p(N, mu, sigma, a, b):
    result = 0
    for i in range(N + 1):
        gam = gamma(i / 2. + 0.5)
        gam_ib = gammainc((b - mu)**2 / (2 * sigma), i / 2. + 0.5)
        gam_ia = gammainc((a - mu)**2 / (2 * sigma), i / 2. + 0.5)
        if (i % 2 == 0):
            incr = 0.5 * gam * (sign(b - mu) * gam_ib - sign(a - mu) * gam_ia)
        else:
            incr = 0.5 * gam * (gam_ib - gam_ia)

        result = result + \
            nchoosek(N, i) * mu**(N - i) * (sqrt(2 * sigma))**(i + 1) * incr

    return result


P = 5
wd = 1
MD = 1
VD = 0.1
eps = 1E-8

binf = (2 - wd) / 2.
bsup = (2 + wd) / 2.

int0 = integral_p(P, MD, VD, binf, bsup)
int1 = integral_p(P + 1, MD, VD, binf, bsup)
int2 = integral_p(P + 2, MD, VD, binf, bsup)

md = int1 / max(int0, eps)
vd = int2 / max(int0, eps) - md**2

print("there must be an error because my variance is negative")
print(md, vd)
