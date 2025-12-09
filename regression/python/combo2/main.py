from math import comb

n = 10
for k in range(n + 1):
    assert comb(n, k) == comb(n, n - k)

