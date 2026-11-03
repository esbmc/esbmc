import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Writer:
    def put(self, w, x):
        w[0] = x
        return len(w)

wr = Writer()
n = wr.put(w=v, x=5)
assert n == 3
assert a[1] == 5
