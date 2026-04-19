def foo(s: str) -> None:
    assert isinstance(s, int), "s must be a string"


foo("foo")
