import math

# NaN self-inequality: complex(nan, 0) != complex(nan, 0)
z_nan = complex(float("nan"), 0.0)
assert not (z_nan == z_nan)
assert (z_nan != z_nan)

# NaN in imag also causes self-inequality.
z_nan_i = complex(0.0, float("nan"))
assert not (z_nan_i == z_nan_i)
assert (z_nan_i != z_nan_i)

# Inf equality: same inf values are equal.
z_inf = complex(float("inf"), 1.0)
z_inf2 = complex(float("inf"), 1.0)
assert z_inf == z_inf2
assert not (z_inf != z_inf2)

# Inf inequality: different sign of inf.
z_neg_inf = complex(float("-inf"), 1.0)
assert z_inf != z_neg_inf
assert not (z_inf == z_neg_inf)

# Cross-type: complex == int when imag is 0.
assert complex(42, 0) == 42
assert 42 == complex(42, 0)
assert not (complex(42, 1) == 42)

# Cross-type: complex == float when imag is 0.
assert complex(3.14, 0) == 3.14
assert 3.14 == complex(3.14, 0)

# Cross-type: complex == bool.
assert complex(1, 0) == True
assert complex(0, 0) == False
assert not (complex(0, 1) == False)

# Complex != non-numeric always.
assert complex(1, 0) != "1"
assert complex(1, 0) != None  # type: ignore
