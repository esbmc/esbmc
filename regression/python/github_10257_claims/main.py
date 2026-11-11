def unused(a: int, b: int) -> int:
    return a // b


def check(x: int) -> None:
    if x < 0:
        raise ValueError("negative")


check(1)
