class Base:
    def __init__(self, n: int) -> None:
        self.n = n

    def twice(self) -> int:
        return 2 * self.n


class Derived(Base):
    pass


ds = [Derived(3), Derived(5)]
assert ds[1].twice() != 10
