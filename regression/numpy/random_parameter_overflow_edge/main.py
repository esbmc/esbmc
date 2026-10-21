import numpy as np

lam_rejected = False
range_rejected = False
try:
    np.random.poisson(1e19)
except ValueError:
    lam_rejected = True
try:
    np.random.uniform(-1e308, 1e308)
except OverflowError:
    range_rejected = True

assert lam_rejected
assert range_rejected
