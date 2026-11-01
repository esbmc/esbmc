# Issue #9445: s[-0:] is the whole string, so this fails for an empty t.
s: str = nondet_str()
t: str = nondet_str()
r: str = s[-len(t):]
assert len(r) < len(s)
