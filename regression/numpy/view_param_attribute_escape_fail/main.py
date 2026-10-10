import numpy as np

class Cache:
    def save(self, row):
        self.cached = row

a = np.array([7, 8, 9])
v = a[0:]
c = Cache()
c.save(v)
