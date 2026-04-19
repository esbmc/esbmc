import math

for n in range(0, 20):
    for k in range(0, n + 1):
        assert math.comb(n, k) == math.comb(n, k)
