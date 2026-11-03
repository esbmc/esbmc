import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Consumer:
    def read(self, w):
        return w[0] + len(w)

c = Consumer()
assert c.read(v) == 3
