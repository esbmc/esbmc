import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Summary:
    def __init__(self, w):
        self.first = w[0]
        self.size = len(w)

s = Summary(v)
assert s.size == 5
