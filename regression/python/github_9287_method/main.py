from mod import obj

if nondet_bool():
    v = nondet_int()
    a = obj.func(v)
    assert a == v
