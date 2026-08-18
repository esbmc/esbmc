# len() on a class instance with no __len__ is a TypeError in CPython, not 0.

class C:
    def __init__(self) -> None:
        self.i: int = 0


def main():
    c = C()
    assert len(c) == 0


main()
