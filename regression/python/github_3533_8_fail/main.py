from typing import Any


def test(flag: int) -> None:
    if flag < 0:
        x: Any = 1
    elif flag == 0:
        x: Any = 1.876
    elif flag >= 1:
        x: Any = False
    y: Any = x
    assert y == 1


test(0)
