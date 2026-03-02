from typing import Any


def test(flag: bool) -> None:
    if flag:
        x: Any = 1
    else:
        x: Any = False
    y: Any = x
    assert y == 1


test(True)
