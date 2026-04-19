def foo(l: list[str]) -> None:
    for s in l:
        assert s == "foo" or s == "bar"

foo(["foo", "bar"])
