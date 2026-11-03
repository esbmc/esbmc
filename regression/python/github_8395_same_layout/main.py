class A:
    def __init__(self) -> None:
        self.x = 1


class B:
    def __init__(self) -> None:
        self.x = 1


assert not isinstance(A(), B)
assert not isinstance(B(), A)
assert isinstance(A(), A)
