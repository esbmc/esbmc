def inner(x: int) -> None:
    assert x != 2


def outer(x: int) -> None:
    inner(x + 1)


outer(1)
