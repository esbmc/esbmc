class A:
    pass


class B(A):
    pass


class C:
    pass


o = B() if nondet_bool() else C()
p = o
if isinstance(p, A):
    assert not isinstance(p, C)
else:
    assert isinstance(o, C)
assert isinstance(p, (A, C))
o = C()
assert isinstance(o, C)
assert not isinstance(o, B)
