def check_zero(x: int | None = None) -> None:
    assert x is not None
    assert x == 0

check_zero(0)
