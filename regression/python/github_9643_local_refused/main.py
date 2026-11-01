import _sv_verifier


class A:
    pass


def f(A):
    x = A
    _sv_verifier.check_type(x, A)


f(3)
