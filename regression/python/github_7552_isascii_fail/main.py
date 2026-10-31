# "aé" is not ASCII, so the assertion fails on the else path (#7552).
c = nondet_bool()
s = "abc" if c else "aé"
assert s.isascii()
