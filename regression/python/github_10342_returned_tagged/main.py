# pick returns a function of helper that returns a str or an int, so r holds
# either, by the path.
from helper import pick

c = nondet_bool()
h = pick(c)
r = h()
assert isinstance(r, str) == c
if c:
    assert r == "a"
    assert len(r) == 1
else:
    assert r + 1 == 2
