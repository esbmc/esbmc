class Box:
    def __init__(self) -> None:
        self.n: int = 7

    def __len__(self) -> int:
        return self.n


def main():
    b = Box()
    assert len(b) == 8


main()
