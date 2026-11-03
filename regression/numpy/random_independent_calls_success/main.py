import numpy as np

a = np.random.randint(0, 2)
b = np.random.randint(0, 2)

assert a + b <= 2
assert a + b >= 0
