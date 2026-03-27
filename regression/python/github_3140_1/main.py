def foo(s: str) -> None:
    assert isinstance(s, str), "s must be a string"

s: str = "a" * True
foo(s)
