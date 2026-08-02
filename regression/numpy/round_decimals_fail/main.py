import numpy as np

# Negative test: np.round(2.567, 1) == 2.6, not 99.0.  ESBMC must detect the
# violated assertion (and must not crash on the 2-argument round form).
assert np.round(2.567, 1) == 99.0
