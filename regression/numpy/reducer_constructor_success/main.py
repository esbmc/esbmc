import numpy as np

# min()/max() over a constructor array, module form.
a = np.eye(3)
assert np.min(a) == 0
assert np.max(a) == 1

# prod() over a constructor array with a fill value, module form.
b = np.full((2, 2), 2)
assert np.prod(b) == 16

# std()/var() over a constructor array, module form.
# identity(2) flattened is [1, 0, 0, 1]; mean = 0.5;
# var = mean((x - mean)^2) = 0.25; std = sqrt(0.25) = 0.5.
c = np.identity(2)
assert np.var(c) == 0.25
assert np.std(c) == 0.5
