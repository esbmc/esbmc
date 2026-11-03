import numpy as np

try:
    np.random.randint(0, 2, dtype=np.complex128)
except TypeError:
    assert False
