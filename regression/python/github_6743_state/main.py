def counted() -> int:
    class Counter:
        def __init__(self) -> None:
            self.n: int = 0

        def bump(self) -> int:
            self.n = self.n + 5
            return self.n

    c = Counter()
    c.bump()
    return c.bump()


def outer() -> int:
    def inner() -> int:
        class Point:
            def value(self) -> int:
                return 4

        return Point().value()

    return inner()


assert counted() == 10
assert outer() == 4
