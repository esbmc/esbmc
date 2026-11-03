from mod import Smth

v = nondet_int()
a = Smth().func(v)
if v == 0:
    assert a == "zero"
    assert isinstance(a, str)
else:
    assert a == v
    assert isinstance(a, int)
