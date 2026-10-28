# KNOWNBUG: a name bound twice has no kind, so k is typed Any (#9277).
n = 1
n = 2
c = nondet_bool()
k = n if c else "a"
assert isinstance(k, str) == (not c)
