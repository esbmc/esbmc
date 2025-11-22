"""
Operational model for non-deterministic list functions in ESBMC Python frontend.

USAGE:
    # Default: list of integers with size in [0, 8]
    x = nondet_list()
    
    # With explicit size:
    x = nondet_list(5)              # list of ints, size in [0, 5]
"""

# Default maximum size for nondet lists
_DEFAULT_NONDET_LIST_SIZE: int = 8


def nondet_list(max_size: int = _DEFAULT_NONDET_LIST_SIZE) -> list:
    """
    Return a non-deterministic list of integers.
    
    Args:
        max_size: Maximum size of the list (default: 8).
                  The actual size will be in range [0, max_size].
    
    Returns:
        list: A list with arbitrary size and integer contents.
    """
    result: list = []
    size: int = nondet_int()
    __ESBMC_assume(size >= 0)
    __ESBMC_assume(size <= max_size)
    
    i: int = 0
    while i < size:
        elem: int = nondet_int()
        result.append(elem)
        i = i + 1
    
    return result
