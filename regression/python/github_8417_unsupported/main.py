def half(v: int) -> float:
    return v / 2


x = nondet_int()
if nondet_bool():
    a = 1
else:
    a = half(x)
if isinstance(a, int):
    assert a == 1
