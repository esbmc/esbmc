def never_called() -> None:
    raise ValueError("never runs")


def check(x: int) -> None:
    if x > 0:
        raise ValueError("positive")


check(1)
