# Negative variant of int_const_numeric (GitHub #4770).
#
# int(65.7) truncates toward zero to 65, so asserting it equals 66 must be
# refuted. Before the fix int(f) folded to 0 for any numeric symbol, which
# would also have refuted this assertion but for the wrong reason; the point
# here is that the conversion now produces the *correct* value 65, not 66.

g: float = 65.7
assert int(g) == 66
