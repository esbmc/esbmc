from typing import Optional

def foo(x: Optional[int]) -> None:
    # These should hold true for proper pointer comparisons
    if x is None:
        assert x == None
        assert not (x != None)
    else:
        assert x != None
        assert not (x == None)

foo(None)
foo(1)
