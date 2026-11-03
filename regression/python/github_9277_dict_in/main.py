# KNOWNBUG: `in` on a tagged key is refused. The key is an int or its str, and the dict holds both.
d = {1: 0, "1": 1}
n = abs(nondet_int()) % 3
c = nondet_bool()
k = n if c else str(n)
if k in d:
    pass
else:
    assert n != 1
