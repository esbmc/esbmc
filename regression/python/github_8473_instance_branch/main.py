def f():
    return 1


class C:
    pass


k = nondet_bool()
if k:
    b = f
else:
    b = C()
if b is f:
    assert k
if b is not f:
    assert not k
