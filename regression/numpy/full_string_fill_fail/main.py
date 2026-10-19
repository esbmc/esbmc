import numpy as np

f = np.full(3, fill_value="x")
assert f[1] == "y"
