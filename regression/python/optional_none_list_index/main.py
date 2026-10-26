from typing import Optional


def read(x: Optional[int]) -> int:
    l = [5, 6]
    try:
        return l[x]
    except TypeError:
        return -1


def store(x: Optional[int]) -> int:
    l = [5, 6]
    try:
        l[x] = 7
    except TypeError:
        return -1
    return l[1]


def clear(x: Optional[int]) -> int:
    l = [5, 6]
    try:
        l[x] = None
    except TypeError:
        return -1
    return 0


def guarded(x: Optional[int]) -> int:
    l = [5, 6]
    if x is not None and l[x] == 6:
        return 1
    return l[x] if x is not None else 9


assert read(None) == -1
assert read(1) == 6
assert store(None) == -1
assert store(1) == 7
assert clear(None) == -1
assert clear(0) == 0
assert guarded(None) == 9
assert guarded(0) == 5
assert guarded(1) == 1
