import _sv_verifier
from mod import smth

v = _sv_verifier.nondet_int()
a = smth.func(v)
assert a == v
assert isinstance(a, int)
