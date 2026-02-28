def foo(l: list[str]) -> None:
    if l is None:
        assert False


foo(["foo"])
