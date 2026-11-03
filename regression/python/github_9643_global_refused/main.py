import _sv_verifier


class A:
    pass


class B:
    pass


def g():
    global A
    A = B


g()
x = B()
_sv_verifier.check_type(x, A)
