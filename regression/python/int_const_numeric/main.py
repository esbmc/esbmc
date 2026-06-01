# Regression test for GitHub #4770.
#
# int() applied to a *constant* numeric variable was mis-routed through the
# string parser (handle_str_symbol_to_int): an int symbol decoded to a single
# character (rejected as non-digit) and a float symbol decoded to nothing, so
# int(x) folded to 0 regardless of x. Numeric symbols must use the numeric
# conversion path instead, which truncates floats toward zero and treats ints
# as identity. Constant strings must keep parsing as before.

f: float = 65.0
assert int(f) == 65

g: float = 65.7
assert int(g) == 65  # truncates toward zero

h: float = -3.9
assert int(h) == -3  # truncation toward zero, not floor

i: int = 42
assert int(i) == 42  # identity on int

s: str = "123"
assert int(s) == 123  # constant string still parses
