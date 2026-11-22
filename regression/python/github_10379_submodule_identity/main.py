import pkg.sub

m = pkg.sub if nondet_bool() else pkg
c = m.func()
if m is pkg:
    assert isinstance(c, float)
    assert c == 2.5
else:
    assert isinstance(c, int)
    assert c == 1
assert (m is not pkg) == (m is pkg.sub)
