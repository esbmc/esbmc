import numpy as np

y = np.expm1(1.0, casting="no")
assert y < 1.0
