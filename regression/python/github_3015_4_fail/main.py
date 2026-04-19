# Test: Unknown keyword argument (should fail)
# Issue #3015: Passing unexpected keyword argument
def foo(x: int, y: str) -> None:
    pass

foo(z=42, x=1, y="foo")

