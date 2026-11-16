# bool() of a str is true exactly when the str is non-empty (#9374).
r = bool("False")
assert r
s = ""
t = bool(s)
assert not t
if not bool("abc"):
    assert False
u = nondet_str()
if len(u) > 0:
    assert bool(u)
else:
    assert not bool(u)
