class A:

    def __init__(self) -> None:
        self.x = 1


class B(A):

    def __init__(self) -> None:
        self.x = 2
        self.y = 3


def is_a(v, h) -> bool:
    return isinstance(v, h)


def is_sub(h) -> bool:
    return issubclass(h, A)


def same(h) -> bool:
    return h is A and h == A


def make(h) -> A:
    return h()


def local_int(v) -> bool:
    t = int
    return isinstance(v, t) and t is int


def local_class() -> A:
    t = B
    return t()


assert is_a(1, int)
assert not is_a("a", int)
assert is_a("a", str)
assert is_a(True, int)
assert is_a(True, bool)
assert not is_a(1, bool)
assert is_a(A(), A)
assert is_a(B(), A)
assert not is_a(A(), B)
assert is_sub(B)
assert not is_sub(int)
assert same(A)
assert not same(B)
a = make(A)
assert a.x == 1
b = make(B)
assert b.x == 2
assert local_int(2)
assert not local_int("b")
c = local_class()
assert isinstance(c, B)
