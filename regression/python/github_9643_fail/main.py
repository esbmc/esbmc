import _sv_verifier


class A:
    pass


class B:
    pass


def f(A):
    return A


x = B()
_sv_verifier.check_type(x, A)
