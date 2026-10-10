class T:
    def __init__(self, n: int) -> None:
        self.n: int = n


def main() -> None:
    t = T(nondet_int())
    d = t.n
    assert getattr(t, "total", d) != 3


main()
