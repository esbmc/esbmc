class C:
    def __init__(self) -> None:
        self.x = 0


c = C()
c.k = 1
assert getattr(c, "k", 2) == 1


def main() -> None:
    d = C()
    d.k = 3
    assert getattr(d, "k") == 3
    assert getattr(d, "x", 5) == 0


main()
