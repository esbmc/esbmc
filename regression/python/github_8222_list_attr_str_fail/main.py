class A:

    def __init__(self) -> None:
        self.names: list[str] = []
        self.vals: list[int] = [7]
        self.names.append("ab")

    def first(self) -> str:
        return self.names[0]


def main() -> None:
    a = A()
    assert a.first() == "ab"
    assert a.vals[0] == 7
    assert a.names[0] == "ac"


main()
