# Test: Correct function call with keyword arguments (should succeed)
# Issue #3015: Correct usage of keyword arguments
def foo(x: int, y: str) -> None:
    assert x == 42
    assert y == "foo"

foo(x=42, y="foo")

