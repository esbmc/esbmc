from typing import Optional

def foo(x: Optional[int]) -> int:
    # This should fail when x is None
    assert x is not None
    return x + 1

foo(None)
