import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD
assert np.exp(1.0) >= 2.718
assert np.exp(0.0) == 1.0
assert np.exp(-1.0) <= 0.368 # e^-1 ≈ 0.3679
assert np.exp(-2.0) <= 0.14  # e^-2 ≈ 0.1353
assert np.exp(-0.1) < 1.0
assert np.exp(-5.0) < 0.01
assert np.exp(-10.0) < 5e-5
assert np.exp(-20.0) < 2.1e-9
<<<<<<< HEAD
=======
assert np.exp(1.0) >= np.e
=======
assert np.exp(1.0) >= 2.718
>>>>>>> 921e0757a ([regression] enabled test case for exp numpy function)
assert np.exp(0.0) == 1.0
>>>>>>> f86e3ad14 ([python] added test cases for numpy math functions (#2437))
=======
>>>>>>> 9adda6dfa ([numpy] enhance numpy math function verification coverage (#2445))
