MyInt = int


def f() -> int:
    x = MyInt(3)
    return x


assert f() == 4
