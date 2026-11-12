# #10311: round() of a float literal beyond 64 bits is not folded to a wrapped int.
a: int = round(-1e300)
assert a == -2147483648
