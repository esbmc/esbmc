import numpy as np
import random

n = random.randint(2, 4)
f = np.full(n, 1)
w = f[0:2]
assert sum(w) == 3
