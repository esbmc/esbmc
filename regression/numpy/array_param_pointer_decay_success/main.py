import numpy as np

# A numpy array is modeled as a fixed-size C array, not the heap PyListObject
# struct backing Python's `list`. Passing it into a user function used to
# silently reinterpret the array's raw bytes as PyListObject fields instead
# of decaying to a compatible element pointer (an unsound pointer-decay bug,
# github next-pr item). It must now fail loudly instead.
def first(a):
    return a[0]

a = np.array([1, 2, 3])
assert first(a) == 1
