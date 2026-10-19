import numpy as np

x = np.random.randint(-5, 5, dtype="int16")
a = np.random.randint(0, 100, size=2, dtype=np.int32)

assert x >= -5
assert x < 5
assert a[1] < 100
