def func() -> int:
    return 2


def add_one(x: int) -> int:
    return x + 1


def power_two(x: int) -> int:
    return x**2


def sum_three(a: int, b: int, c: int) -> int:
    return a + b + c


assert sum_three(power_two(func()), func(), add_one(func())) == 1
