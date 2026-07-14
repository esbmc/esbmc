import numpy as np

s = np.sin(-0.0)
a = np.arctan(-0.0)

assert s == 0.0
assert a <= 0.0
