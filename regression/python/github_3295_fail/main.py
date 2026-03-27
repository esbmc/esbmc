def f(s: str) -> None:
    ss: str = s[2:]
    assert ss == "oo"


s: str = "foo"
f(s)
