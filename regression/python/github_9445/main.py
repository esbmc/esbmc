# Issue #9445: a slice bound negating a call result.
s: str = nondet_str()
t: str = nondet_str()
r: str = s[-len(t):]
if len(t) == 0:
    assert r == s
elif len(t) < len(s):
    assert len(r) == len(t)
else:
    assert r == s
u: str = s[:-len(t)]
assert len(u) <= len(s)
