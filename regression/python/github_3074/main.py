def foo(x: str | None = None) -> None:
    if x is not None:
        assert len(x) > 0


foo("foo")
