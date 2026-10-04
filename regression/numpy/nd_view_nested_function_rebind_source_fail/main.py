import numpy as np


def outer():
    def inner():
        b = np.array([[1, 2], [3, 4]])
        v = b[:, 0]
        b = np.array([[7, 7], [7, 7]])
        assert v[1] == 7

    inner()


outer()
