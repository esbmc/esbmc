import _sv_verifier


class A:
    pass


def g():
    global A
    A = 3


x = A()
_sv_verifier.check_type(x, A)
