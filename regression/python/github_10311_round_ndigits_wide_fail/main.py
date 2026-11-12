# #10311: an ndigits literal beyond int range is not truncated.
a = round(2.5, 4294967296)
assert a == 2.0
