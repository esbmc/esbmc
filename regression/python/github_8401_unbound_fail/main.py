class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


def f(c: bool) -> None:
    if c:
        a = A()
    assert type(a) is A


f(False)
