import numpy as np

a = np.random.randint(-3, 3, size=3, dtype=np.int16)

assert a.shape == (3,)
assert a[0] >= -3
assert a[2] < 3
