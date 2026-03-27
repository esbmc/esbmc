def foo(x: int | None = None) -> None:
    if x is not None:
        assert x == 42


foo(5)
