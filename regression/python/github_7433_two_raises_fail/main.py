def check(x: int) -> None:
    if x > 0:
        raise ValueError("positive")
    raise ValueError("non-positive")


check(1)
