# Test: Unknown keyword argument (should fail)
# Issue #3015: Passing unexpected keyword argument
def foo(y: int, x: str) -> None:
    pass


z: int = 42
foo(z, y=1, x="foo")
