class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


def f(x: A) -> None:
    assert type(x) is A


f(B())
