from mod import func

if nondet_bool():
    v = nondet_int()
    a = func(v)
    assert a != 0
