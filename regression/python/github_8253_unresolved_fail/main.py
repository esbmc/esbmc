class A:
    def __init__(self) -> None:
        self.total: int = 1


class B:
    def __init__(self) -> None:
        self.n: int = 1


b = B()
assert getattr(b, "total", 1) is None
