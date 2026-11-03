# str.isascii() on a string only known at run time (#7552).
c = nondet_bool()
s = "abc" if c else "aé"
assert s.isascii() == c
assert s[0].isascii()
