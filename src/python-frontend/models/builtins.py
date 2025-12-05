# def abs(x:float) -> float:
#     if x >= 0:
#         return x
#     else:
#         return -x


def all(iterable:list[Any]) -> bool:
    """Return True if all elements of the iterable are true (or if empty)."""
    i: int = 0
    length: int = len(iterable)
    while i < length:
        element:bool = iterable[i]
        if not element:
            return False
        i = i + 1
    return True


def any(iterable: list[Any]) -> bool:
    """Return True if any element of the iterable is true."""
    i: int = 0
    length: int = len(iterable)
    while i < length:
        element:bool = iterable[i]
        if element:
            return True
        i = i + 1
    return False
