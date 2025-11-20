LL: list[str] = ["foo", "foobar"]

def foo(s: str) -> None:
    assert s in LL


foo("foobar")
