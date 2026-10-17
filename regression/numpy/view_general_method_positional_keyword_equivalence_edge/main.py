import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Consumer:
    def read(self, w, k):
        return w[k] + len(w)

c = Consumer()
p = c.read(v, 1)
q = c.read(w=v, k=1)
r = c.read(v, k=1)
assert p == q
assert q == r
assert p == 7
