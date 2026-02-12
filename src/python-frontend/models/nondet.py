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

from typing import Any

# Shared default maximum size for nondet collections
_DEFAULT_NONDET_SIZE: int = 8


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


def nondet_list(max_size: int = _DEFAULT_NONDET_SIZE, elem_type: Any = None) -> list:
    """
    Return a non-deterministic list with specified element type.

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
    result: list = []
    size: int = _nondet_size(max_size)
    i: int = 0
    while i < size:
        if elem_type is None:
            e: Any = nondet_int()
            result.append(e)
        else:
            e: Any = elem_type
            result.append(elem_type)
        i = i + 1

    return result


def nondet_dict(max_size: int = _DEFAULT_NONDET_SIZE,
                key_type: Any = None,
                value_type: Any = None) -> dict:
    """
    Return a non-deterministic dictionary with specified key and value types.

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
    result: dict = {}
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        # Generate new key each iteration
        if key_type is None:
            k: int = nondet_int()
        else:
            k1: Any = key_type
        # TODO here we should do like
        # elif isinstance(key_type, bool):
        #   k_bool: Any = nondet_bool()
        # elif isinstance(key_type, float):
        #   k_float: Any = nondet_float()
        # but for now we dont support isinstance and int/bool... keys,
        # so we just return key_type directly if it is not None

        # Generate new value each iteration
        # TODO same as key_type
        if value_type is None:
            v: int = nondet_int()
        else:
            v1: Any = value_type

        if key_type is None:
            if value_type is None:
                result[k] = v
            else:
                result[k] = v1
        else:
            if value_type is None:
                result[k1] = v
            else:
                result[k1] = v1
        i = i + 1

    return result