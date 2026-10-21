import numpy as np

a = np.arange(6)
v = a[1:4]
x = np.random.choice(v)

assert x >= 1
assert x <= 3
