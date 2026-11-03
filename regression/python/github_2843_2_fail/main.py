from typing import Any

def foo(x: int) -> Any:
    if x == 4:
        return True
    else:
        return 5

if nondet_bool():
    assert foo(4) != True
if nondet_bool():
    assert foo(0) != 5
