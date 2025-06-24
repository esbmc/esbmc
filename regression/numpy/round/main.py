import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD
# Basic rounding
=======
>>>>>>> f86e3ad14 ([python] added test cases for numpy math functions (#2437))
=======
# Basic rounding
>>>>>>> 9adda6dfa ([numpy] enhance numpy math function verification coverage (#2445))
assert np.round(1.2) == 1.0
assert np.round(1.8) == 2.0
assert np.round(-1.5) == -2.0
a = np.round(-1.8)
assert a == -2.0
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 9adda6dfa ([numpy] enhance numpy math function verification coverage (#2445))

# Very small numbers
assert np.round(0.0000001) == 0.0
assert np.round(-0.0000001) == 0.0

# Large numbers
assert np.round(1234567.89) == 1234568.0
assert np.round(-1234567.89) == -1234568.0

# Already rounded
assert np.round(10.0) == 10.0
assert np.round(-10.0) == -10.0

<<<<<<< HEAD
=======
>>>>>>> f86e3ad14 ([python] added test cases for numpy math functions (#2437))
=======
>>>>>>> 9adda6dfa ([numpy] enhance numpy math function verification coverage (#2445))
