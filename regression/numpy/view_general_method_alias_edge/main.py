import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Writer:
    def put(self, w):
        w[0] = 8
        return w[0]

b = v
wr = Writer()
x = wr.put(b)
assert x == 8
assert v[0] == 8
assert a[1] == 8
