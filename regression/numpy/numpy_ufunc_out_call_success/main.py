import numpy as np

r = np.exp([0.0, 0.0], out=np.empty(2))
assert r[0] == 1.0 and r[1] == 1.0
s = np.hypot([3.0], [4.0], out=np.zeros(1))
assert s[0] >= 4.0 and s[0] <= 5.7
