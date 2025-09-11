def func() -> int:
    return 2


def add_one(x: int) -> int:
    return x + 1


def multiply_two(x: int) -> int:
    return x * 2


def complex_calc(a: int, b: int, c: int) -> int:
    return a * b + c


assert complex_calc(func(), add_one(func()), multiply_two(func())) == 10
