import numpy as np

a = np.random.choice([8], size=(1,))

assert a.shape == (1,)
assert a[0] == 8
