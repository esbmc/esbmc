import numpy as np

b = np.arange(10)
v = b[::3]
a = np.random.choice(v, 2)

assert a[0] % 3 == 0
assert a[1] <= 9
