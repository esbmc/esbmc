class A:
    pass


class B:
    pass


o = A() if nondet_bool() else B()
if isinstance(o, A):
    pass
else:
    assert isinstance(o, B)
