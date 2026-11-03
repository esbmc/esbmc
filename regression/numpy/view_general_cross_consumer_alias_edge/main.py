import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Writer:
    def put(self, w):
        w[0] = 6
        return len(w)

def total(w):
    t = sum(w)
    return t

b = v
n = Writer().put(b)
assert total(v) == 11
assert max(b) == 6
assert a[1] == 6
