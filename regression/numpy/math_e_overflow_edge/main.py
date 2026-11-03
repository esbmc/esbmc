import numpy as np

assert np.exp(709.78) < float("inf")
assert np.exp(709.79) == float("inf")
