def func() -> int:
    return 2


def increment(x: int) -> int:
    return x + 1


assert increment(increment(increment(func()))) == 5
