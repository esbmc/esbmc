def foo(s: str) -> None:
    l: list[str] = ['a', 'b', 'c']
    for ss in l:
        assert ss not in s

foo("foo")
