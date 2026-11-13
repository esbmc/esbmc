class A:
    pass


class A2:
    pass


c = nondet_bool()
if c:
    a = A()
else:
    a = A2
if c:
    assert isinstance(a, A)
else:
    assert not isinstance(a, A)
