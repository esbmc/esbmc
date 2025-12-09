def foo(x: int | None = None) -> None:
    if x is not None and x is not None:
        assert False

foo()
