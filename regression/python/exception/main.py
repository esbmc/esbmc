def foo(value: int) -> int:
    if value < 0:
        raise ValueError("Negative value!")

    return value * 2


result = 1

try:
    result = foo(-1)
except ValueError as e:
    print(e)

assert result == 1
