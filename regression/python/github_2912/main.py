def foo(s: str) -> None:
    if s != "foo":
        raise ValueError("Unexpected value '{s}' passed to foo()")


foo("foo")
