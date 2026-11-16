from mod import func

if nondet_bool():
    v = nondet_int()
    a = func(v)
    assert a == v
else:
    a = func()
    assert a == "a"
