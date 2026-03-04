from typing import Any

def test(flag: bool) -> None:
    if flag:
        x: Any = 1
    else:
        x: Any = 1.876
    y: Any = x
    assert y == False

test(False)