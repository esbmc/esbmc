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


def sorted(iterable: list[int]) -> list[int]:
    """Return a new sorted list from the items in iterable."""
    # Create a copy of the list
    result: list[int] = []
    i: int = 0
    length: int = len(iterable)

    # Copy all elements
    while i < length:
        result.append(iterable[i])
        i = i + 1

    # Bubble sort (simple and verifier-friendly)
    n: int = len(result)
    i = 0
    while i < n:
        j: int = 0
        while j < n - 1 - i:
            if result[j] > result[j + 1]:
                # Swap
                temp: int = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp
            j = j + 1
        i = i + 1

    return result


def sorted_float(iterable: list[float]) -> list[float]:
    """Return a new sorted list from the items in iterable."""
    result: list[float] = []
    i: int = 0
    length: int = len(iterable)

    while i < length:
        result.append(iterable[i])
        i = i + 1

    n: int = len(result)
    i = 0
    while i < n:
        j: int = 0
        while j < n - 1 - i:
            if result[j] > result[j + 1]:
                temp: float = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp
            j = j + 1
        i = i + 1

    return result


def sorted_str(iterable: list[str]) -> list[str]:
    """Return a new sorted list from the items in iterable."""
    result: list[str] = []
    i: int = 0
    length: int = len(iterable)

    while i < length:
        result.append(iterable[i])
        i = i + 1

    n: int = len(result)
    i = 0
    while i < n:
        j: int = 0
        while j < n - 1 - i:
            if result[j] > result[j + 1]:
                temp: str = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp
            j = j + 1
        i = i + 1

    return result


def zip(a: str, b: str) -> list:
    """Minimal zip support for two strings."""
    result: list = []
    i: int = 0
    len_a: int = len(a)
    len_b: int = len(b)
    n: int = len_a if len_a < len_b else len_b

    while i < n:
        result.append((a[i], b[i]))
        i = i + 1

    return result
