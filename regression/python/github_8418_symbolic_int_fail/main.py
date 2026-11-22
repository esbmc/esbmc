x = nondet_int()
if nondet_bool():
    a = x
    t = a + 1
    assert t == x
else:
    a = 0.5
