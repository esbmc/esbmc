import numpy as np

# Read-only access through a 1-D slice view must already work correctly
# whether the view is a copy (current behavior) or a real alias (this PR):
# nothing mutates between creation and read, so both representations agree.
a = np.array([1, 2, 3, 4])
v = a[1:3]

assert v[0] == 2
assert v[1] == 3
assert a[1] == 2
assert a[2] == 3
