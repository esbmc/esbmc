def func() -> int:
    return 2


def id(x: int) -> int:
    return x


assert id(func()) == 1  # This should fail
