class C:
    def __init__(self) -> None:
        self.i: int = 0


def main():
    c = C()
    caught: int = 0
    try:
        n: int = len(c)
    except TypeError:
        caught = 1
    assert caught == 1


main()
