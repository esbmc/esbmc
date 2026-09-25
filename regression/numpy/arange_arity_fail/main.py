import numpy as np

# Wrong argument counts must raise the specific arity message rather than
# falling through to the generic "non-constant inputs" TypeError.
a = np.arange(1, 2, 3, 4)
assert len(a) == 0
