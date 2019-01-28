from math import exp, sqrt, erf, erfc, pi
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


# Routines
def Gauss(x):
    return exp(-x * x / 2) / sqrt(2 * pi)


def new_mx(mhat, alpha, rhoX, f_x):
    def f_to_int(x):
        return Gauss(x) * \
            (f_x(1. / (alpha * mhat), 0 + x / sqrt((alpha * mhat)), rhoX)) ** 2
    (int1, err1) = quad(f_to_int, -10, 10)
    int2 = 0
    if (rhoX > 0.001):
        def g_to_int(x):
            return (Gauss(x) *
                    (f_x(1. / (alpha * mhat), x *
                         sqrt(1 + 1. / (alpha * mhat)), rhoX))**2)
        (int2, err2) = quad(g_to_int, -10, 10)
    return (1 - rhoX) * int1 + (rhoX) * int2


def H_(x):
    return 0.5 * erfc(x / sqrt(2))


def gout(w, Y, V, theta=0):
    DELTA = 0.0001
    bottom_line = (1e-10 + sqrt(2 * pi * (DELTA + V)) *
                   H_(-(2 * Y - 1) * (w - theta) / sqrt(DELTA + V))
                   )
    top_line = ((2 * Y - 1) *
                exp(-0.5 * ((w - theta)**2 / (DELTA + V)))
                )
    return top_line / bottom_line


def dgout(w, Y, V, theta=0):
    DELTA = 0.0001
    g = gout(w, Y, V, theta)
    return -max(g * ((w - theta) / (DELTA + V) + g), 1e-10)


def new_mhat(mx, Z02, theta):
    def f(x):
        return (Gauss(x) *
                dgout(x * sqrt(mx), 0, Z02 - mx, theta) *
                erfc(x * sqrt(mx) / sqrt(2 * (Z02 - mx))) * 0.5)

    (int1, err1) = quad(f, -10, 10)

    def g(x):
        return (Gauss(x) *
                dgout(x * sqrt(mx), 1, Z02 - mx, theta) *
                erfc(- x * sqrt(mx) / sqrt(2 * (Z02 - mx))) * 0.5)

    (int2, err2) = quad(g, -10, 10)
    return -(int1 + int2)


def f_gaussbernoulli(S2, R, rho=0.5, m=0, s2=1):
    Z = (1 - rho) * \
        exp(-R * R / (2 * S2)) \
        + rho * sqrt(S2 / (S2 + s2)) * exp(-((R - m)**2) / (2 * (S2 + s2)))
    UP2 = rho * (1 - rho) \
        * exp(- R * R / (2 * S2) - ((R - m)**2) / (2 * (S2 + s2))) \
        * (sqrt(S2) / (S2 + s2)**(2.5)) \
        * (s2 * S2 * (S2 + s2) + (m * S2 + R * s2)**2)\
        + rho * rho * exp(-((R - m)**2) / ((S2 + s2))) \
        * (s2 * S2**2) / (s2 + S2)**2
    UP1 = rho * exp(-((R - m)**2) / (2 * (S2 + s2)))\
        * (sqrt(S2) / (S2 + s2)**(1.5)) * (m * S2 + R * s2)
    F_a = UP1 / Z
    F_b = UP2 / Z**2
    return F_a, F_b


def perform_DE(mxstart, rhoX, alpha, f_x, theta=0, criterion=1e-6, tmax=1000):
    # First compute Z02 and init values
    Z02 = rhoX
    mx = mxstart - 1e-6
    diff = 1
    t = 0
    mhat = 0
    while ((diff > criterion and t < tmax)):
        mhat = new_mhat(mx, Z02, theta)
        t = t + 1
        mx_new = 0.5 * new_mx(mhat, alpha, rhoX, f_x) + 0.5 * mx
        diff = abs(mx_new - mx)
        mx = mx_new
        if (abs(Z02 - mx) < criterion):
            break
    return Z02 - mx, mx, t


def compute_MSE_range_alpha(rhoX, rangealpha, f_x, theta=0):
    valMSEX = np.zeros(rangealpha.size)
    valM = np.zeros(rangealpha.size)
    valt = np.zeros(rangealpha.size)
    mxstart = 0.01
    print("alpha, M, t GEN")
    for j in np.arange(1, rangealpha.size, 1):
        (MSEX, M, t) = perform_DE(mxstart, rhoX, rangealpha[j], f_x, theta)
        valMSEX[j] = MSEX
        valM[j] = M
        valt[j] = t
        mxstart = M
        print(rangealpha[j], M, t, generalization(M, rhoX))
    return valMSEX, valM, valt


def generalization(mx, rho):
    if mx < 1e-6:
        mx = 1e-3

    V = rho - mx
    if (V < 1e-6):
        V = 1e-6

    def f(x):
        return Gauss(x) * erf(x * sqrt(mx / (2 * V)))**2

    (int1, err1) = quad(f, -10, 10)
    return 1 - int1


theta = 0
rhoX = 1


def f_x(x, y, z):
    return f_gaussbernoulli(x, y, z, 0, 1)[0]


rangealpha = np.arange(3, 100, 1)
(X1, M1, T1) = compute_MSE_range_alpha(rhoX, rangealpha, f_x, theta)

