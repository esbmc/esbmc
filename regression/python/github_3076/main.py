def foo(s: str) -> str:
    return s

s: str = foo("foo")
assert len(s) == 3
