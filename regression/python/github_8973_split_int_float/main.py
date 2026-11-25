x = nondet_int()
if nondet_bool():
    a = x
else:
    a = float(x % 1000)
r = a + 1
assert type(r) is type(a)
if isinstance(a, int):
    assert r == x + 1
else:
    assert 1 <= r < 1001
