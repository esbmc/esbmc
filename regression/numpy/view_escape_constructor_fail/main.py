import numpy as np


class Holder:
    def __init__(self, v):
        self.view = v


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
h = Holder(row)
