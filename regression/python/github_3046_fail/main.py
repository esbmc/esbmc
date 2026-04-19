def foo(l: list[str]) -> None:
    for s in l:
        assert s == "foo"

foo(["foo", "bar"])
