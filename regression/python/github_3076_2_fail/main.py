def foo() -> str:
    return "foo"

s: str = foo()
assert len(s) == 4
