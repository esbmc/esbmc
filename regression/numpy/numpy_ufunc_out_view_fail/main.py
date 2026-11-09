import numpy as np

c = np.zeros(3)
np.sqrt(np.array([4.0, 9.0]), out=c[1:])
assert c[2] == 0.0
