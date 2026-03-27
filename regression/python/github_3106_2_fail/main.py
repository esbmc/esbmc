def foo(a: bytes | str | None = None) -> None:
    if a is not None:
        assert False


a = "a"
foo(a=a)
