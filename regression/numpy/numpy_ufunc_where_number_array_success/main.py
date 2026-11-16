import numpy as np

o = np.zeros(2)
np.exp([0.0, 0.0], where=[1, 0], out=o)
assert o[0] == 1.0 and o[1] == 0.0
p = np.zeros(2)
np.exp([0.0, 0.0], where=np.array([False, True]), out=p)
assert p[0] == 0.0 and p[1] == 1.0
