import numpy as np


# A non-constant argument (e.g. sourced from a function parameter) cannot be
# fast-path materialized. Falling back to the operational model's while-loop
# implementation used to hang past any practical timeout instead of
# producing a verdict, so this must be rejected explicitly and quickly
# instead.
def f(n):
    return np.arange(n)


a = f(5)
assert len(a) == 5
