class Base:
    def __init__(self) -> None:
        self.n: int = 5

    def __len__(self) -> int:
        return self.n


class Derived(Base):
    pass


def main():
    d = Derived()
    assert len(d) == 5


main()
