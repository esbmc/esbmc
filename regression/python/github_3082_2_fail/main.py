def foo(s: str) -> None:
    ss = s[1:3]
    assert ss == "foo"

foo("foobar")
