class A:
    pass


class B:
    pass


o = A() if nondet_bool() else B()
assert isinstance(o, A)
