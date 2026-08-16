import numpy as np


def read_row():
    a = np.array([[1, 2], [3, 4]])
    row = a[0]
    assert len(row) == 2
    return 2


assert read_row() == 2
