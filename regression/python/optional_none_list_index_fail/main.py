from typing import Optional


def read(x: Optional[int]) -> int:
    l = [5, 6]
    try:
        return l[x]
    except TypeError:
        return -1


assert read(1) == 6
assert read(None) == 5
