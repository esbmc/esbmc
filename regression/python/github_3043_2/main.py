def foo(s: str) -> None:
    l = 2
    ss: str = s[1:l]
    assert ss == "a"

foo("bar")
