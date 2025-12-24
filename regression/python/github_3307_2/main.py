def foo(s: list[int] | None = None) -> None:
    if s is not None:
        assert len(s) > 0, "s must not be empty"

foo([1, 2, 3])
