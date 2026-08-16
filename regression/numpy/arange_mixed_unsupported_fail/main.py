import numpy as np


# A non-constant float argument (e.g. sourced from a function parameter)
# cannot be fast-path materialized either, matching the int case; it is
# rejected explicitly and quickly instead of falling back to the
# operational model, which hangs past any practical timeout.
def f(x):
    return np.arange(x)


a = f(2.5)
assert len(a) == 3
