class A:

    def __init__(self) -> None:
        self.x = 1


class B(A):

    def __init__(self) -> None:
        self.x = 2
        self.y = 3


def is_a(v, h) -> bool:
    return isinstance(v, h)


assert is_a(B(), A)
assert is_a(A(), B)
