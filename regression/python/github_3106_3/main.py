def foo(a: bytes | str | None = None) -> None:
    if a is None:
        assert False


foo(72)
