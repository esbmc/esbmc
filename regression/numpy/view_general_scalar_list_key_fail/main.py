import numpy as np
import random

r = random.randint(0, 1) == 1
a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert max(v[0], v[1], key=lambda x: -x) == 4
