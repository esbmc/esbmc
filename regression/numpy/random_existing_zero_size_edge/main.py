import numpy as np

a = np.random.randint(0, 3, size=0)

assert len(a) == 0
assert a.shape == (0,)
