import numpy as np

# np.eye/identity/linspace results were not tracked in numpy_array_symbols_,
# so a method call on one (e.g. a.sum()) silently stopped being rewritten
# into the np.sum(a) dispatch form instead of raising a diagnostic.
a = np.eye(2)
s = a.sum()
assert s == 2.0

c = np.identity(3)
s2 = c.sum()
assert s2 == 3.0

e = np.linspace(0.0, 4.0, 5)
g = e.copy()
assert g[0] == 0.0
assert g[4] == 4.0
