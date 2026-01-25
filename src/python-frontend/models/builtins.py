# def abs(x:float) -> float:
#     if x >= 0:
#         return x
#     else:
#         return -x


def all(iterable: list[Any]) -> bool:
    """Return True if all elements of the iterable are true (or if empty)."""
    i: int = 0
    length: int = len(iterable)
    while i < length:
        element: bool = iterable[i]
        if not element:
            return False
        i = i + 1
    return True

# NOTE:
# These implementations intentionally duplicate code instead of using helpers.
# Flat functions generate simpler GOTO/SSA and verify faster in ESBMC.
# Do not refactor into higher-order or generic helpers.

def max(iterable: list[int]) -> int:
    """Return the maximum element from an iterable of integers."""
    if len(iterable) == 0:
        raise ValueError("max() arg is an empty sequence")

    i: int = 1
    length: int = len(iterable)
    result: int = iterable[0]

    while i < length:
        element: int = iterable[i]
        if element > result:
            result = element
        i = i + 1

    return result


def max_float(iterable: list[float]) -> float:
    """Return the maximum element from an iterable of floats."""
    if len(iterable) == 0:
        raise ValueError("max() arg is an empty sequence")

    i: int = 1
    length: int = len(iterable)
    result: float = iterable[0]

    while i < length:
        element: float = iterable[i]
        if element > result:
            result = element
        i = i + 1

    return result


def max_str(iterable: list[str]) -> str:
    """Return the maximum element from an iterable of strings."""
    if len(iterable) == 0:
        raise ValueError("max() arg is an empty sequence")

    i: int = 1
    length: int = len(iterable)
    result: str = iterable[0]

    while i < length:
        element: str = iterable[i]
        if element > result:
            result = element
        i = i + 1

    return result


def min(iterable: list[int]) -> int:
    """Return the minimum element from an iterable of integers."""
    if len(iterable) == 0:
        raise ValueError("min() arg is an empty sequence")

    i: int = 1
    length: int = len(iterable)
    result: int = iterable[0]

    while i < length:
        element: int = iterable[i]
        if element < result:
            result = element
        i = i + 1

    return result


def min_float(iterable: list[float]) -> float:
    """Return the minimum element from an iterable of floats."""
    if len(iterable) == 0:
        raise ValueError("min() arg is an empty sequence")

    i: int = 1
    length: int = len(iterable)
    result: float = iterable[0]

    while i < length:
        element: float = iterable[i]
        if element < result:
            result = element
        i = i + 1

    return result


def min_str(iterable: list[str]) -> str:
    """Return the minimum element from an iterable of strings."""
    if len(iterable) == 0:
        raise ValueError("min() arg is an empty sequence")

    i: int = 1
    length: int = len(iterable)
    result: str = iterable[0]

    while i < length:
        element: str = iterable[i]
        if element < result:
            result = element
        i = i + 1

    return result


# def any(iterable: list[Any]) -> bool:
#     """Return True if any element of the iterable is true."""
#     i: int = 0
#     length: int = len(iterable)
#     while i < length:
#         element: bool = iterable[i]
#         if element:
#             return True
#         i = i + 1
#     return False
