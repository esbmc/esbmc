class Foo:

    def __init__(self, a: int) -> None:
        self.a = a

    def same(self, b: int) -> bool:
        return self.a == b


def main() -> None:
    f = Foo(1)
    f.same(b=1)  # <------ The usage of a keyword argument here seems to be the issue


if __name__ == "__main__":
    main()
