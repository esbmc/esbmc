import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Consumer:
    def first(self, w):
        return w[0] + len(w)

r = a[::-2]
c = Consumer()
assert c.first(r) == 8
