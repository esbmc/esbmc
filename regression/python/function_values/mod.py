def f1() -> int:
    return 1


def f2() -> int:
    return 2


def pick(c: bool):
    if c:
        return f1
    return f2
