import numpy as np

a = np.random.randint(0, 3, size=2)

try:
    a[2]
except IndexError:
    assert False
