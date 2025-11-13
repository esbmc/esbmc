def foo(x: int | None = None) -> None:
    if x is not None:
        assert len(x) == 0

foo()

