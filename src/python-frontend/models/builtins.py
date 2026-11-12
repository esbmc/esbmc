# pylint: disable=redefined-builtin,undefined-variable
# These functions intentionally shadow Python built-ins and reference
# typing forward-declarations (e.g. `Any`) and ESBMC intrinsics that
# have no Python binding: they are the operational models ESBMC uses
# to verify Python programs, so they must match the built-in names
# exactly.
#
# pylint: disable=consider-using-max-builtin,consider-using-min-builtin
# This module is itself the source of max() / min() for the verification
# model. Rewriting the explicit branch forms here as max()/min() would
# create a self-reference: the model would call into itself.


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


def max_default(iterable: list[int], default: int) -> int:
    """Return the max of an iterable of ints, or default if empty."""
    if len(iterable) == 0:
        return default

    i: int = 1
    length: int = len(iterable)
    result: int = iterable[0]

    while i < length:
        element: int = iterable[i]
        if element > result:
            result = element
        i = i + 1

    return result


def max_default_float(iterable: list[float], default: float) -> float:
    """Return the max of an iterable of floats, or default if empty."""
    if len(iterable) == 0:
        return default

    i: int = 1
    length: int = len(iterable)
    result: float = iterable[0]

    while i < length:
        element: float = iterable[i]
        if element > result:
            result = element
        i = i + 1

    return result


def max_default_str(iterable: list[str], default: str) -> str:
    """Return the max of an iterable of strings, or default if empty."""
    if len(iterable) == 0:
        return default

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


def any(iterable: list[Any]) -> bool:
    """Return True if any element of the iterable is true."""
    i: int = 0
    length: int = len(iterable)
    while i < length:
        element: bool = iterable[i]
        if element:
            return True
        i = i + 1
    return False


def min_default(iterable: list[int], default: int) -> int:
    """Return the min of an iterable of ints, or default if empty."""
    if len(iterable) == 0:
        return default

    i: int = 1
    length: int = len(iterable)
    result: int = iterable[0]

    while i < length:
        element: int = iterable[i]
        if element < result:
            result = element
        i = i + 1

    return result


def min_default_float(iterable: list[float], default: float) -> float:
    """Return the min of an iterable of floats, or default if empty."""
    if len(iterable) == 0:
        return default

    i: int = 1
    length: int = len(iterable)
    result: float = iterable[0]

    while i < length:
        element: float = iterable[i]
        if element < result:
            result = element
        i = i + 1

    return result


def min_default_str(iterable: list[str], default: str) -> str:
    """Return the min of an iterable of strings, or default if empty."""
    if len(iterable) == 0:
        return default

    i: int = 1
    length: int = len(iterable)
    result: str = iterable[0]

    while i < length:
        element: str = iterable[i]
        if element < result:
            result = element
        i = i + 1

    return result


def sum(iterable: list[int], start: int = 0) -> int:
    """Return start plus the sum of all elements in an iterable of integers."""
    result: int = start
    i: int = 0
    length: int = len(iterable)
    while i < length:
        element: int = iterable[i]
        result = result + element
        i = i + 1
    return result


def sum_float(iterable: list[float], start: float = 0.0) -> float:
    """Return start plus the sum of all elements in an iterable of floats."""
    result: float = start
    i: int = 0
    length: int = len(iterable)
    while i < length:
        element: float = iterable[i]
        result = result + element
        i = i + 1
    return result


def sorted(iterable: list[int], reverse: bool = False) -> list[int]:
    """Return a new sorted list from the items in iterable."""
    # Create a copy of the list
    result: list[int] = []
    i: int = 0
    length: int = len(iterable)

    # Copy all elements
    while i < length:
        result.append(iterable[i])
        i = i + 1

    # Python's sort is stable for reverse=True as well: reverse, sort
    # ascending, reverse back, as CPython does, so equal elements keep their
    # order. list.sort() lowers to the insertion sort of the C list model.
    if reverse:
        result.reverse()
    result.sort()
    if reverse:
        result.reverse()

    return result


def sorted_float(iterable: list[float], reverse: bool = False) -> list[float]:
    """Return a new sorted list from the items in iterable."""
    result: list[float] = []
    i: int = 0
    length: int = len(iterable)

    while i < length:
        result.append(iterable[i])
        i = i + 1

    if reverse:
        result.reverse()
    result.sort()
    if reverse:
        result.reverse()

    return result


def sorted_str(iterable: list[str], reverse: bool = False) -> list[str]:
    """Return a new sorted list from the items in iterable."""
    result: list[str] = []
    i: int = 0
    length: int = len(iterable)

    while i < length:
        result.append(iterable[i])
        i = i + 1

    if reverse:
        result.reverse()
    result.sort()
    if reverse:
        result.reverse()

    return result


def reversed(iterable: list[int]) -> list[int]:
    """Return a new list with the items of iterable in reverse order.

    CPython's reversed() yields a reverse iterator; modelling it as a freshly
    built list is sound in every list-consuming context ESBMC handles (slice
    assignment, list(), iteration), mirroring how sorted() returns a list.
    """
    result: list[int] = []
    i: int = len(iterable) - 1
    while i >= 0:
        result.append(iterable[i])
        i = i - 1
    return result


def reversed_float(iterable: list[float]) -> list[float]:
    """Return a new list with the items of iterable in reverse order."""
    result: list[float] = []
    i: int = len(iterable) - 1
    while i >= 0:
        result.append(iterable[i])
        i = i - 1
    return result


def reversed_str(iterable: list[str]) -> list[str]:
    """Return a new list with the items of iterable in reverse order."""
    result: list[str] = []
    i: int = len(iterable) - 1
    while i >= 0:
        result.append(iterable[i])
        i = i - 1
    return result


def _starred_arity_error() -> int:
    # The frontend calls this when a starred argument does not fill the
    # callee's remaining parameters.
    raise TypeError("starred argument does not match the parameter count")


def _call_type_error(message: str) -> int:
    # The frontend calls this in place of a call whose arguments do not bind.
    raise TypeError(message)
