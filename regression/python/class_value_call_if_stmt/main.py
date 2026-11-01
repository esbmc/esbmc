class A:
    pass


class B:
    pass


c = nondet_bool()
if c:
    cls = A
else:
    cls = B
o = cls()
assert isinstance(o, A) == c
assert isinstance(o, B) != c
