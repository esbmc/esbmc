def foo(l: list[str]) -> None:
    ll = ["foo", "bar", "baz"]
    for s in l:
        assert s in ll


foo(["foo", "bar"])
