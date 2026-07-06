# A dict with mixed int/float values now reads every value correctly. The
# subscript read used one static value type (from the first value), so it
# misread whichever value carried the other type; e.g. {0: 1, 1: 2.5}[1]
# read 2.5's bits through the int path. Values are read by their runtime
# type_id instead. Covers int-first, float-first, three values, string
# keys, and a value assigned via d[k]=v.
d = {0: 1, 1: 2.5}
assert d[0] == 1
assert d[1] == 2.5
e = {0: 2.5, 1: 1}
assert e[0] == 2.5
assert e[1] == 1
f = {0: 1, 1: 2.5, 2: 3}
assert f[1] == 2.5 and f[2] == 3
g = {"a": 1, "b": 2.5}
assert g["b"] == 2.5
h = {0: 1}
h[1] = 2.5
assert h[1] == 2.5 and h[0] == 1
# A >32-bit int value must survive the widened read (regresses reading it as
# a 32-bit long under LLP64/ILP32 instead of int64).
k = {0: 5000000000, 1: 2.5}
assert k[0] == 5000000000
