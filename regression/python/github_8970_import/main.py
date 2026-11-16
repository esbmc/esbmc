# Two imported functions of different return types, chosen by nondet_bool().
from helper import f, g

c = nondet_bool()
h = f if c else g
r = h()
if c:
    assert isinstance(r, str)
    assert r == "a"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
assert (h is f) == c
