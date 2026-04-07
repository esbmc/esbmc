"""
Operational model for non-deterministic collection functions in ESBMC Python frontend.

USAGE:
    # Lists:
    x = nondet_list()                                    # int list, size [0, 8]
    x = nondet_list(5)                                   # int list, size [0, 5]
    x = nondet_list(elem_type=nondet_float())                 # float list, size [0, 8]
    x = nondet_list(max_size=10, elem_type=nondet_bool())     # bool list, size [0, 10]

    # Dictionaries:
    d = nondet_dict()                                    # int->int dict, size [0, 8]
    d = nondet_dict(5)                                   # int->int dict, size [0, 5]
    d = nondet_dict(key_type=nondet_str(), value_type=nondet_float())
    d = nondet_dict(max_size=10, key_type=nondet_int(), value_type=nondet_bool())
"""

# Shared default maximum size for nondet collections
_DEFAULT_NONDET_SIZE: int = 8

# Type flags (concrete constants, resolved before any loop)
_T_INT: int = 0
_T_FLOAT: int = 1
_T_BOOL: int = 2
_T_STR: int = 3


def _nondet_size(max_size: int) -> int:
    """
    Generate a non-deterministic size in range [0, max_size].

    Args:
        max_size: Maximum size (inclusive).

    Returns:
        int: A non-deterministic integer in [0, max_size].
    """
    size: int = nondet_int()
    __ESBMC_assume(size >= 0)
    __ESBMC_assume(size <= max_size)
    return size


def _type_flag(sample) -> int:
    """Determine a concrete type flag from a sample nondet value.
    Evaluated once before the loop; the result is a plain int constant."""
    if isinstance(sample, float):
        return _T_FLOAT
    if isinstance(sample, bool):
        return _T_BOOL
    if isinstance(sample, str):
        return _T_STR
    return _T_INT


def nondet_list(max_size: int = _DEFAULT_NONDET_SIZE, elem_type=None) -> list:
    """
    Return a non-deterministic list where each element is a fresh nondet value.

    Args:
        max_size: Maximum size of the list (default: 8).
                  The actual size will be in range [0, max_size].
        elem_type: Value returned by type constructor for list elements (default: nondet_int()).
                   Supported: nondet_int(), nondet_float(), nondet_bool(), nondet_str()

    Returns:
        list: A list with arbitrary size and contents of specified type.

    Examples:
        x = nondet_list()                                    # int list, size [0, 8]
        x = nondet_list(5)                                   # int list, size [0, 5]
        x = nondet_list(elem_type=nondet_float())            # float list, size [0, 8]
        x = nondet_list(max_size=10, elem_type=nondet_bool())# bool list, size [0, 10]
    """
    tf: int = _type_flag(elem_type)

    result: list = []
    size: int = _nondet_size(max_size)

    # Each branch has its own loop with a single nondet call — no branching inside the loop.
    i: int = 0
    if tf == _T_FLOAT:
        while i < size:
            result.append(nondet_float())
            i = i + 1
    elif tf == _T_BOOL:
        while i < size:
            result.append(nondet_bool())
            i = i + 1
    elif tf == _T_STR:
        while i < size:
            result.append(nondet_str())
            i = i + 1
    else:
        while i < size:
            result.append(nondet_int())
            i = i + 1

    return result


def nondet_dict(max_size: int = _DEFAULT_NONDET_SIZE,
                key_type=None,
                value_type=None) -> dict:
    """
    Return a non-deterministic dictionary where each entry has fresh nondet key and value.

    Args:
        max_size: Maximum size of the dictionary (default: 8).
                  The actual size will be in range [0, max_size].
        key_type: Value returned by type constructor for dictionary keys (default: nondet_int()).
                  Supported: nondet_int(), nondet_str(), nondet_bool()
        value_type: Value returned by type constructor for dictionary values (default: nondet_int()).
                    Supported: nondet_int(), nondet_float(), nondet_bool(), nondet_str()

    Returns:
        dict: A dictionary with arbitrary size and contents of specified types.

    Examples:
        d = nondet_dict()                    # int->int dict, size [0, 8]
        d = nondet_dict(5)                   # int->int dict, size [0, 5]
        d = nondet_dict(key_type=nondet_str(), value_type=nondet_float())
        d = nondet_dict(max_size=10, key_type=nondet_int(), value_type=nondet_bool())
    """
    kt: int = _type_flag(key_type)
    vt: int = _type_flag(value_type)
    result: dict = {}
    size: int = _nondet_size(max_size)

    # The flag comparisons (kt == X, vt == X) use concrete int constants,
    # so dead branches are trivially eliminated — only the matching
    # nondet call survives in the unrolled loop body.
    i: int = 0
    while i < size:
        if kt == _T_STR:
            k = nondet_str()
        elif kt == _T_BOOL:
            k = nondet_bool()
        else:
            k = nondet_int()

        if vt == _T_FLOAT:
            v = nondet_float()
        elif vt == _T_BOOL:
            v = nondet_bool()
        elif vt == _T_STR:
            v = nondet_str()
        else:
            v = nondet_int()

        result[k] = v
        i = i + 1

    return result
