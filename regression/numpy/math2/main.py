import numpy as np

assert np.fabs(-2.1) == 2.1
assert np.fabs(2.1) == 2.1
assert np.fabs(-3) == 3 
assert np.fabs(3) == 3
fa = np.fabs(-3.2)
assert fa == 3.2
