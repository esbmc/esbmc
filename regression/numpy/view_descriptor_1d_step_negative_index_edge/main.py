import numpy as np

# Guard for the upcoming stepped-slice pointer view: negative indexing into
# the view itself (not just a negative slice bound) must still resolve
# against the view's own logical length once it becomes a strided pointer.
a = np.array([1, 2, 3, 4, 5])
v = a[::2]

assert len(v) == 3
assert v[-1] == a[4]
assert v[-2] == a[2]
