import numpy as np

r = np.log(np.e)
assert r > 0.9999999999999 and r < 1.0000000000001
assert np.log(1.0) == 0.0
