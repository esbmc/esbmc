def func() -> int:
    return 2


def add_one(x: int) -> int:
    return x + 1


def multiply_two(x: int) -> int:
    return x * 2


assert add_one(multiply_two(func())) == 5
