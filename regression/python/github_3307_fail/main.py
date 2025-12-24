def foo(s: list[str] | None = None) -> None:
    if s is not None:
        assert len(s) == 4, "s has three elements"

foo(["a", "b", "c"])
