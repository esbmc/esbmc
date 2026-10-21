import numpy as np

a = np.random.choice([7], size=(2, 0))
b = np.random.choice((), size=0)

assert a.shape == (2, 0)
assert len(b) == 0
