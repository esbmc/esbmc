# #10311: round() of a float literal does not wrap at 32 bits.
a: int = round(3e9)
assert a < 0
