class C:
    def __init__(self) -> None:
        self.i: int = 0


def main():
    for v in C():
        pass
    assert True


main()
