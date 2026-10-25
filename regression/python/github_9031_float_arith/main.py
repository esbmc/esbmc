c = nondet_bool()
if c:
    r = "a"
else:
    r = 1.5
if isinstance(r, float):
    assert r + 1 == 2.5
