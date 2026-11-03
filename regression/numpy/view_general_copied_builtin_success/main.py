import numpy as np
import random

r = np.arange(5, dtype=int)
w = r[1:]
assert sum(w) == 10
assert min(w) == 1
assert max(w) == 4
assert all(w)
assert tuple(w)[0] == 1
assert sorted(w, reverse=True)[0] == 4
n = random.randint(2, 4)
f = np.full(n, 1)
u = f[0:2]
assert sum(u) == 2
assert r[1] == 1
