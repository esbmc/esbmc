# Two different classes bound as values are not equal (#7549).
class C:
    pass


class D:
    pass


class Holder:
    t = int


def main() -> None:
    x = C
    assert x == D


main()
