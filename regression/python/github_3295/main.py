def f(s: str) -> None:
    ss: str = s[1:]
    assert ss == "oo"

s: str = "foo"
f(s)
