import numpy as np


# Plain lists, not numpy views: a function may return a slice of its own
# list parameter from any path, here through recursion.
def first_of(items):
    if len(items) <= 1:
        return items
    return first_of(items[:1])


a = np.array([1, 2, 3])
assert first_of([30, 20, 10]) == [30]
assert a[0] == 1
