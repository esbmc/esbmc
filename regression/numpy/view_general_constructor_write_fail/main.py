import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Marker:
    def __init__(self, w, x):
        w[0] = x
        self.x = x

m = Marker(v, 7)
assert a[1] == 1
