import numpy as np

r = np.hypot(1.7976931348623157e308, 1.0)
assert r != 1.7976931348623157e308
