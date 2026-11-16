class A:
    pass


class B:
    pass


cls = A
o = cls()
assert isinstance(o, A)
cls = B
p = cls()
assert isinstance(p, B)
