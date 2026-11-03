import numpy as np
import random

r = random.randint(0, 1) == 1
a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
t = sorted(v, key=lambda x: -x, reverse=r)
assert t[0] == 4
