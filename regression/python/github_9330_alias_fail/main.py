class A:
    pass


class B:
    pass


Alias = A
o = A() if nondet_bool() else B()
if isinstance(o, Alias):
    assert False
