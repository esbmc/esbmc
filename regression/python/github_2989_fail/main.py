def foo(s: str | None = None) -> None:
    if s is not None:
        assert isinstance(s, None)

foo("foo")
