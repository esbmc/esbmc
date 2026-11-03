class Inner:
    def __init__(self) -> None:
        self.n: int = 1


class T:
    def __init__(self) -> None:
        self.inner = Inner()


t = T()
assert getattr(t.inner, "n") == 1
assert getattr(t, "total", None) is None
