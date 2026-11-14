def f(x: int) -> None:
    assert type(max(x, 0.5)) is int


f(1)
