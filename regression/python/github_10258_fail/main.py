def f(n: int) -> int:
    try:
        return n // 0
    except ZeroDivisionError:
        return -1


def g(n: int) -> int:
    return n % (2 - 2)


assert f(3) == -1
g(5)
