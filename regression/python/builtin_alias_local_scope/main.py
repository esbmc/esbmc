def f() -> int:
    MyInt = int
    return MyInt(3)


assert f() == 3
