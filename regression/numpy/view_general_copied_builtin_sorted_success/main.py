import numpy as np

r = np.arange(5, dtype=int)
w = r[1:]
assert sorted(w, reverse=True)[0] == 4
assert r[1] == 1
