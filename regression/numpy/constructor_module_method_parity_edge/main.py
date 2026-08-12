import numpy as np

# Module-form and method-form are exercised on two separate receivers,
# each built by a different constructor, to confirm each call resolves
# its own receiver rather than assuming a shared/aliased array.
a = np.zeros((2, 2))
b = np.full((2, 2), 5)

assert np.sum(a) == 0
assert b.sum() == 20
