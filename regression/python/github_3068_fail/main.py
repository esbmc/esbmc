def foo(x: str) -> None:
    assert not x.endswith("foo")


foo("foo")
