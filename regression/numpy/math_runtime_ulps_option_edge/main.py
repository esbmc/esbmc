import numpy as np

d = np.exp(1.0) - np.e
assert d > -4.5e-16 and d < 4.5e-16
