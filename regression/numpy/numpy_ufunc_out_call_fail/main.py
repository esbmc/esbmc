import numpy as np

r = np.exp([0.0, 0.0], out=np.empty(2))
assert r[1] == 0.0
