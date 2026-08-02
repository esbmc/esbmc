assert (2.5).as_integer_ratio() == (5, 2)
assert (0.5).as_integer_ratio() == (1, 2)
assert (-2.5).as_integer_ratio() == (-5, 2)
assert (3.0).as_integer_ratio() == (3, 1)
assert (0.0).as_integer_ratio() == (0, 1)
assert (5).as_integer_ratio() == (5, 1)
assert (-7).as_integer_ratio() == (-7, 1)
# Exact dyadic ratio of a non-representable decimal.
assert (0.1).as_integer_ratio() == (3602879701896397, 36028797018963968)
# Tuple unpacking of the folded result.
n, d = (1.5).as_integer_ratio()
assert n == 3 and d == 2
