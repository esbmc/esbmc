def foo(x: int | None = None) -> None:
    if x is not None:
        assert False

foo()
