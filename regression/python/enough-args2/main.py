class Foo:
    def __init__(self, a: int) -> None:
        self.a = a
        assert self.a == a

    def same(self, b: int) -> bool:
        result = self.a == b
        assert result == (self.a == b)
        return result

def main() -> None:
        f = Foo(1)
        assert f.a == 1

        r = f.same(b=1)
        assert r is True
        assert f.a == 1

if __name__ == "__main__":
    main()

