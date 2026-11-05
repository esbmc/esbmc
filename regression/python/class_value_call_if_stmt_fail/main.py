class A:
    pass


class B:
    pass


if nondet_bool():
    cls = A
else:
    cls = B
o = cls()
assert isinstance(o, A)
