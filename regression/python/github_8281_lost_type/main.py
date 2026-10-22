import _sv_verifier

n = _sv_verifier.nondet_int()


def g():
    return n


def func(a):
    return a()


b = func(g)
func(lambda: n)
_sv_verifier.check_type(b, int)
