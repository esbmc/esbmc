class A:
    def __init__(self) -> None:
        self.v = 1


def f(x) -> bool:
    return isinstance(x, tuple)


assert f((1, 2))
assert not f(A())
