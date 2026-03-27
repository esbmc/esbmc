# Test: Unknown keyword argument (should fail)
# Issue #3015: Passing unexpected keyword argument
def foo(x: int, y: str) -> None:
    pass


z: int = 42
foo(z, x=1, y="foo")
