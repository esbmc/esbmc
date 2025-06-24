import numpy as np

assert np.arctan(0.0) >= 0.0
assert np.arctan(1.0) >= (3.14153 / 4)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 9adda6dfa ([numpy] enhance numpy math function verification coverage (#2445))

# arctan(1) ≈ π/4 ≈ 0.785
x = np.arctan(1.0) - (3.14 / 4)
assert abs(x) < 1e-2

# arctan(-1) ≈ -π/4 ≈ -0.785
y = np.arctan(-1.0) + (3.14 / 4)
assert abs(y) < 1e-2

# arctan(√3) ≈ π/3 ≈ 1.047
z = np.arctan(1.73) - (3.14 / 3)
assert abs(z) < 0.05

# arctan(-√3) ≈ -π/3 ≈ -1.047
x = np.arctan(-1.73) + (3.14 / 3)
assert abs(x) < 0.05

# arctan(∞) ≈ π/2 ≈ 1.57
y = np.arctan(1e6) - (3.14 / 2)
assert abs(y) < 0.01

# arctan(-∞) ≈ -π/2 ≈ -1.57
z = np.arctan(-1e6) + (3.14 / 2)
assert abs(z) < 0.01
<<<<<<< HEAD
=======
>>>>>>> f86e3ad14 ([python] added test cases for numpy math functions (#2437))
=======
>>>>>>> 9adda6dfa ([numpy] enhance numpy math function verification coverage (#2445))
