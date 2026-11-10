# `l.pick` and `right.pick` share a name and return different types.
import left as l
import right

c = nondet_bool()
h = l.pick if c else right.pick
r = h(41)
if c:
    assert r == 42
else:
    assert r is None
