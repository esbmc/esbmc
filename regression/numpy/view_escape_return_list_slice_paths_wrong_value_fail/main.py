import numpy as np


def first_of(items):
    if len(items) <= 1:
        return items
    return first_of(items[:1])


a = np.array([1, 2, 3])
assert first_of([30, 20, 10]) == [20]
