import numpy as np

try:
    np.random.randint(0, 300, size=2, dtype=np.int8)
except ValueError:
    assert False
