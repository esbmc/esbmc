# Dict comprehension whose generator target is a tuple, unpacking each
# (key, value) element of a list of tuples. The frontend previously rejected
# this with "Only simple targets are supported in DictComp"; it now unpacks the
# element tuple into the component names via the shared tuple-unpacking path.

pairs = [(1, 2), (3, 4)]

d = {v: u for u, v in pairs}
assert d[2] == 1
assert d[4] == 3

# Both component names are usable in the key/value expressions.
e = {u: u + v for u, v in pairs}
assert e[1] == 3
assert e[3] == 7
