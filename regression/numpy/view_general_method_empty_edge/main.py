import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Consumer:
    def size(self, w):
        n = len(w)
        return n

e = a[2:2]
c = Consumer()
assert c.size(e) == 0
