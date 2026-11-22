import random
import _sv_verifier


class A:
    pass


a = A()
code = 0
if random.random() < 0.5:
    code = 1
    a.code = code
_sv_verifier.check_type(a, A)
assert getattr(a, "code", 0) == code
