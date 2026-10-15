def double(f):

    def wrapper(x: int) -> int:
        return 2 * f(x)

    return wrapper


def register(f):
    return f


def other(x: int) -> int:
    return x + 10


def replace(f):
    return other


def unrelated(f):

    def inner(x):
        return "s"

    return inner


@double
def g(x: int) -> int:
    return x


@double
def h(x: int) -> int:
    return x + 1


@register
def k(x: int) -> int:
    return x + 1


@replace
def m(x: int) -> int:
    return x


@unrelated
def func(x):
    return x


def p(x: int) -> int:
    return x


@double
@double
def dd(x: int) -> int:
    return x


def triple(f):

    def wrapper(x: int) -> int:
        return 3 * f(x)

    return wrapper


@triple
def t(x: int) -> int:
    return x


def triple(f):
    return f


q = double(p)

# A failing assert ends its path, so each check runs on a path of its own.
n = nondet_int()
if n == 0:
    assert g(3) == 3
elif n == 1:
    assert h(3) == 6
elif n == 2:
    assert k(1) == 1
elif n == 3:
    assert m(1) == 1
elif n == 4:
    assert func(1) == 1
elif n == 5:
    assert q(3) == 3
elif n == 6:
    assert dd(3) == 6
else:
    assert t(3) == 3
