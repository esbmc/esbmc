import numpy as np

a = np.linspace(0.0, 4.0, 5)
b = a.flatten()
assert b[4] == 4.0

# flatten's result must stay independent of the source: mutating b does not
# change a.
b[0] = 99.0
assert a[0] == 0.0
