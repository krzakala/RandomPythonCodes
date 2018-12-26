from scipy.misc import comb


def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n / k * nchoosek(n - 1, k - 1)
    return round(r)


print(nchoosek(3, 2))
print(comb(3, 2))



