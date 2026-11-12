k = nondet_int()
if k == 0:
    a = "Hello"
elif k == 1:
    a = 5
if k == 0:
    assert a == "Hello"
else:
    assert isinstance(a, int)
