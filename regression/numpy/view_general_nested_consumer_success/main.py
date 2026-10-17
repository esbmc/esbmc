import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]

class Consumer:
    def total(self, w):
        return sum(w) + max(w)

c = Consumer()
assert c.total(v) == 10
assert sum(list(v)) == 6
assert min(sorted(v)) == 1
