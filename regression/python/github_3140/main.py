def foo(s: str) -> None:
    assert isinstance(s, str), "s must be a string"

s: str = "a" * 3
foo(s)
