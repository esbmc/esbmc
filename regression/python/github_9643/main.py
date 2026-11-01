import _sv_verifier


class A:
    def __init__(self, x=1):
        pass


def f(A):
    x = A()
    y = A(1)


z = f(A)
x = A()
y = A(1)
_sv_verifier.check_type(A, type)
_sv_verifier.check_type(x, A)
_sv_verifier.check_type(y, A)
_sv_verifier.check_type(z, type(None))
