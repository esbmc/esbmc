def foo(s: list[int] | None = None) -> None:
    if s is not None:
        assert len(s) == 2, "s has three elements"


foo([1, 2, 3])
