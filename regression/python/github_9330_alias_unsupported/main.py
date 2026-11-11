class A:
    pass


class B:
    pass


Alias = A
c = nondet_bool()
o = A() if c else B()
if isinstance(o, (Alias, B)):
    assert not c
