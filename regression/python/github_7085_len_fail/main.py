class C:
    def __init__(self) -> None:
        self.i: int = 0


def main():
    c = C()
    assert len(c) == 0


main()
