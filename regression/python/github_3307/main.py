def foo(s: list[str] | None = None) -> None:
    if s is not None:
        assert len(s) > 0, "s must not be empty"

foo(["a", "b", "c"])
