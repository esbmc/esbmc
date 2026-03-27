# Test: Multiple values for keyword argument (should fail)
# Issue #3015: foo gets multiple values for argument "x"
def foo(x: int, y: str) -> None:
    pass


foo(42, x=1, y="foo")
