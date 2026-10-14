import numpy as np

r = np.sqrt([[0.0, 1.0], [4.0, 9.0]])

assert r[0][0] == 0.0
assert r[0][1] == 1.0
assert r[1][0] == 2.0
assert r[1][1] == 3.0
