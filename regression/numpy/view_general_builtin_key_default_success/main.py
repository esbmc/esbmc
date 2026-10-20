import numpy as np
import random

r = random.randint(0, 1) == 1
a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert max(v, key=lambda x: -x, default=0) == 1
assert min(v, key=lambda x: -x, default=0) == 4
w = a[4:1]
assert max(w, key=lambda x: -x, default=9) == 9
