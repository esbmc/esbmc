def foo(l: list[str] | None = None) -> None:
    if l is not None:
        for s in l:
            assert s == "foo"


foo(["foo", "bar"])
