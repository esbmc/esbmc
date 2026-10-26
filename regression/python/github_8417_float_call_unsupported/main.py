x = nondet_int()
if nondet_bool():
    a = 1
else:
    a = float(x % 10)
if isinstance(a, int):
    assert a == 1
