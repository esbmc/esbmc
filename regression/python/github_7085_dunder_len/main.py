# A class that does define __len__ still dispatches to it.

class C:
    def __init__(self) -> None:
        self.n: int = 5

    def __len__(self) -> int:
        return self.n


def main():
    c = C()
    assert len(c) == 5


main()
