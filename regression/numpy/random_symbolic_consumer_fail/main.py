import numpy as np

a = np.random.random(3)
i = np.random.randint(3, 4)

try:
    a[i]
except IndexError:
    assert False
