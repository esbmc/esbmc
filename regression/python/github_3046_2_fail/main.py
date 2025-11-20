def foo(l: list[str]) -> None:
    if l is not None:
        assert False

foo(["foo"])
