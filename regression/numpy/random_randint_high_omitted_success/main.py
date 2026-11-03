import numpy as np

a = np.random.randint(5, size=3)

assert a[0] >= 0
assert a[1] < 5
assert a[2] < 5
