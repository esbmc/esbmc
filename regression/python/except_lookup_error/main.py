# KeyError is a LookupError, so the handler catches it.
d = {1: 2}
try:
    v = d[3]
except LookupError:
    v = 0
assert v == 0
