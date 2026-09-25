import numpy as np


class Box:
    pass


a = np.array([[1, 2], [3, 4]])
row = a[0]
box = Box()

box.value = row
