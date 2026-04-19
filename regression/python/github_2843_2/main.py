from typing import Any

def foo(x: int) -> Any:
    if x == 4:
        return True
    else:
        return 5

assert foo(4) == True
assert foo(0) == 5
