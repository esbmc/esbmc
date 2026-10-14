class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


a = A()
b = B()
assert type(a) is A
assert type(a) == A
assert type(a) is not B
assert type(b) is B
assert type(b) is not A
assert type(b) != A
assert type(1) is int


def f() -> None:
    c = B()
    assert type(c) is B
    assert type(c) is not A


f()
