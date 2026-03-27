def foo(s: str | None = None) -> None:
    l = ['a', 'b', 'c']
    if s is not None:
        for ss in l:
            assert ss not in s


foo("")
