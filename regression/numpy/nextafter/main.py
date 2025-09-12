import numpy as np

assert np.nextafter(1.0, 2.0) > 1.0
assert np.nextafter(1.0, 0.0) < 1.0
