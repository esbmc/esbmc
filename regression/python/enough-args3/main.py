class Foo:

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b
        assert self.a == a
        assert self.b == b

    def total(self) -> int:
        return self.a + self.b


def main() -> None:
    f = Foo(1, b=2)  # mixed positional + keyword args
    result = f.total()
    assert result == 3


if __name__ == "__main__":
    main()
