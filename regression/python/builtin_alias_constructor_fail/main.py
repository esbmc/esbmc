MyInt = int


def f(x: int) -> int:
    return MyInt(x) + 1


assert f(4) == 6
