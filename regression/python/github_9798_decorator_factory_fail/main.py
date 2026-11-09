def times(k):

    def dec(f):

        def w(x: int) -> int:
            return k * f(x)

        return w

    return dec


def plus(n=1):

    def dec(f):

        def w(x: int) -> int:
            return f(x) + n

        return w

    return dec


c = 3


@times(c)
def g(x: int) -> int:
    return x + 1


@plus(n=10)
@times(2)
def h(x: int) -> int:
    return x


@plus()
def p(x: int) -> int:
    return x


@times(4)
@times(5)
def q(x: int) -> int:
    return x


c = 100
assert g(1) == 6
assert h(4) == 19
assert p(4) == 5
assert q(1) == 20
