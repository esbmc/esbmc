# square root
assert abs(4 ** 0.5 - 2.0) < 1e-9
# cube root
assert abs(8 ** (1/3) - 2.0) < 1e-9
# fractional exponent > 1
assert abs(27 ** (2/3) - 9.0) < 1e-9
# negative base with integer exponent
assert (-4) ** 3 == -64
# ensure float result for non-integer exponent
res = 9 ** 0.5
assert isinstance(res, float)

