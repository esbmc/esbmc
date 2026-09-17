class Seq:
    def __init__(self) -> None:
        self.n: int = 3

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        if i >= self.n:
            raise IndexError
        return i * 2


def main() -> None:
    total: int = 0
    for v in Seq():
        total += v
    assert total == 7


main()
