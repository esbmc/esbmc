c = nondet_bool()
if c:
    r = "a"
else:
    r = 1.5
assert isinstance(r, str) == c
assert isinstance(r, float) != c
assert not isinstance(r, int)
