class A(Exception):
    pass


class B:
    pass


o = A() if nondet_bool() else B()
assert isinstance(o, Exception) or isinstance(o, B)
