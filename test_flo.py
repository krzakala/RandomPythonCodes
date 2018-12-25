from math import exp, sqrt, erf, erfc, pi
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


# Routines
def Stability(rho, K=1):
    A = 4 * K * K * exp(- K * K / rho) / (2 * pi * rho)
    B = (1. / erf(K / sqrt(2 * rho)) + 1. / erfc(K / sqrt(2 * rho)))
    return A * B


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


def gout(w, Y, V, theta=1):
    V = V + 1e-6
    A = ((2 * Y) / (sqrt(2 * pi * V)))
    B = exp(-(theta**2 + w**2) / (2 * V) - theta * w / V) \
        * (exp(2 * theta * w / V) - 1)
    if (w > 0):
        B = exp(-(theta**2 + w**2) / (2 * V) + theta * w / V) \
            * (1 - exp(-2 * theta * w / V))
    C = 1E-5 + \
        erfc(-Y * (theta + w) / (sqrt(2 * V)))\
        - Y * erfc((theta - w) / (sqrt(2 * V)))
    return A * B / C


def new_mhat(mx, Z02, theta=1):
    V_eff = max(Z02 - mx, 1e-5)
    mx = mx + 1e-5

    def g(x):
        return (gout(x * sqrt(mx), 1, V_eff, theta)**2 *
                (1 - 0.5 * erfc((theta + x * sqrt(mx)) / sqrt(2 * V_eff)) -
                 0.5 * erfc((theta - x * sqrt(mx)) / sqrt(2 * V_eff))) +
                (gout(x * sqrt(mx), -1, V_eff, theta)**2) *
                (0.5 * erfc((theta + x * sqrt(mx)) / sqrt(2 * V_eff)) + 0.5 *
                 erfc((theta - x * sqrt(mx)) / sqrt(2 * V_eff)))
                )

    def f(x):
        return Gauss(x) * g(x)
    (int1, err1) = quad(f, -5, 5)
    return (int1)


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
    print("alpha, M, t")
    for j in np.arange(1, rangealpha.size, 1):
        (MSEX, M, t) = perform_DE(mxstart, rhoX, rangealpha[j], f_x, theta)
        valMSEX[j] = MSEX
        valM[j] = M
        valt[j] = t
        mxstart = M
        print(rangealpha[j], M, t)
    return valMSEX, valM, valt


theta = 0.674489
rhoX = 1
alpha_C = 1. / Stability(rhoX, theta)


def f_x(x, y, z):
    return f_gaussbernoulli(x, y, z, 0, 1)[0]


rangealpha = np.arange(0.01, 2, 0.01)
(X1, M1, T1) = compute_MSE_range_alpha(rhoX, rangealpha, f_x, theta)
rangealpha2 = np.arange(2, 0.01, -0.01)
(X2, M2, T2) = compute_MSE_range_alpha(rhoX, rangealpha2, f_x, theta)

plt.subplot(1, 3, 1)
plt.plot(rangealpha, M1, 'b*')
plt.plot(rangealpha2, M2, 'r-')
plt.axvline(x=alpha_C, color='g')
plt.ylabel('overlap')
plt.xlabel('alpha')
plt.subplot(1, 3, 2)
plt.plot(rangealpha, T1, 'b*')
plt.plot(rangealpha2, T2, 'r-')
plt.axvline(x=alpha_C, color='g')
plt.ylabel('iteration time')
plt.xlabel('alpha')
plt.subplot(1, 3, 3)
plt.plot(rangealpha, X1, 'b*')
plt.plot(rangealpha2, X2, 'r-')
plt.axvline(x=alpha_C, color='g')
plt.ylabel('MSE')
plt.xlabel('alpha')
plt.show()
