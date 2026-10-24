import numpy as np

# NumPy's exp(0.3) is 1.3498588075760032; the C model is not.
assert np.exp(0.3) != 1.3498588075760032
