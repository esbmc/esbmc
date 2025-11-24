"""
Operational model for non-deterministic list functions in ESBMC Python frontend.

USAGE:
    # Default: list of integers with size in [0, 8]
    x = nondet_list()
    
    # With explicit size:
    x = nondet_list(5)              # list of ints, size in [0, 5]

    # With type specification:
    x = nondet_list(type=nondet_float())  # list of floats
    x = nondet_list(type=nondet_bool())   # list of bools
    x = nondet_list(max_size=10, type=nondet_int())  # list of ints, size in [0, 10]
"""

from typing import Any

# Default maximum size for nondet lists
_DEFAULT_NONDET_LIST_SIZE: int = 8


def nondet_list(max_size: int = _DEFAULT_NONDET_LIST_SIZE, nondet_type: Any = None) -> list:
    """
    Return a non-deterministic list with specified element type.
    
    Args:
        max_size: Maximum size of the list (default: 8).
                  The actual size will be in range [0, max_size].
        nondet_type: Type constructor for list elements (default: nondet_int()).
              Supported: nondet_int(), nondet_float(), nondet_bool()
    
    Returns:
        list: A list with arbitrary size and contents of specified type.

    Examples:
        >>> x = nondet_list()                          # int list, size [0, 8]
        >>> y = nondet_list(5)                         # int list, size [0, 5]
        >>> z = nondet_list(type=nondet_float())         # float list, size [0, 8]
        >>> w = nondet_list(max_size=10, type=nondet_bool())  # bool list, size [0, 10]
    """
    # Default to nondet_int if no type specified
    if nondet_type is None:
        nondet_type = nondet_int()

    result: list = []
    size: int = nondet_int()
    __ESBMC_assume(size >= 0)
    __ESBMC_assume(size <= max_size)

    i: int = 0
    while i < size:
        elem: Any = nondet_type
        result.append(elem)
        i = i + 1

    return result
