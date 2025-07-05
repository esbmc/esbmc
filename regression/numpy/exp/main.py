import numpy as np

assert np.exp(1.0) >= 2.718
assert np.exp(0.0) == 1.0
assert np.exp(-1.0) <= 0.368 # e^-1 ≈ 0.3679
assert np.exp(-2.0) <= 0.14  # e^-2 ≈ 0.1353
assert np.exp(-0.1) < 1.0
assert np.exp(-5.0) < 0.01
assert np.exp(-10.0) < 5e-5
assert np.exp(-20.0) < 2.1e-9