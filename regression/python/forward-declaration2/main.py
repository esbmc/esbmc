def f1() -> int:
    return f2() + f3()

def f2() -> int:
    return 10

def f3() -> int:
    return 20

assert f1() == 30

