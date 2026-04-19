from typing import Any

def foo(x: int) -> Any:
    if x == 4:
        return True
    elif x == 2:
        return "Any"
    else:
        return 5

assert foo(4) == True
assert foo(0) == 5
assert foo(2) == "Any"
