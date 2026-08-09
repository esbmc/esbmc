import numpy as np

# min()/max() over a constructor array, module and method form.
a = np.eye(3)
assert np.min(a) == 0
assert a.min() == 0
assert np.max(a) == 1
assert a.max() == 1

# prod() over a constructor array with a fill value, module form.
b = np.full((2, 2), 2)
assert np.prod(b) == 16

# std()/var() over a constructor array, module and method form.
# var = mean((x - mean)^2) over [1,0,0,1] with mean 0.5 = 0.1875;
# std = sqrt(0.1875) = 0.4330127018922193.
c = np.identity(2)
assert np.var(c) == 0.1875
assert c.var() == 0.1875
std_module = np.std(c)
assert std_module > 0.4330126 and std_module < 0.4330128
std_method = c.std()
assert std_method > 0.4330126 and std_method < 0.4330128
