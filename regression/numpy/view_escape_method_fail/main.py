import numpy as np


class Store:
    def __init__(self):
        self.n = 0

    def put(self, v):
        self.n = self.n + 1


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
s = Store()
s.put(row)
