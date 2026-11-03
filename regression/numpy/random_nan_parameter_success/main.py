import numpy as np

nan = float("nan")
x = np.random.normal(0.0, nan)
assert x != x
b = np.random.beta(nan, 1.0)
assert b != b
c = np.random.chisquare(nan)
assert c != c
g = np.random.gamma(nan)
assert g != g
m = np.random.lognormal(nan, 1.0)
assert m != m
assert np.random.poisson(-0.0) == 0
assert np.random.binomial(1, -0.0) == 0
