class Res:

    def __init__(self, n: int) -> None:
        self.n = n

    def get(self) -> int:
        return self.n


class Mgr:

    def __init__(self, n: int) -> None:
        self.n = n
        self.closed = False

    def __enter__(self) -> Res:
        return Res(self.n)

    def __exit__(self, a: object, b: object, c: object) -> bool:
        self.closed = True
        return False


def acquire(n: int) -> Mgr:
    return Mgr(n)
