def func() -> int:
    return 2


def add_one(x: int) -> int:
    return x + 1


def add(x: int, y: int) -> int:
    return x + y


def subtract(x: int, y: int) -> int:
    return x - y


def multiply_two(x: int) -> int:
    return x * 2


assert add(func(), add_one(func())) == 5
assert subtract(multiply_two(func()), func()) == 2
