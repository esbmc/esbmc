"""
Operational model for non-deterministic collection functions in ESBMC Python frontend.

KNOWN LIMITATIONS:
(why we implemented mothod in preprocessor instead of here)

  The ideal fix for both nondet_list and nondet_dict is simple:
  call nondet_int()/nondet_float()/etc. fresh inside the loop instead of
  reusing a single pre-evaluated value.
  For example:

      # Ideal nondet_list:
      while i < size:
          result.append(nondet_int())   # fresh value each iteration

      # Ideal nondet_dict:
      while i < size:
          result[nondet_int()] = nondet_int()   # fresh key and value

  However, these fixes cannot be implemented in this model file due to
  some ESBMC frontend limitations(so far we know):

    nondet_list:  adding if/elif branches with different result.append()
      types (e.g. nondet_int() vs nondet_float()) causes
      "unresolved operand type" errors when accessing list elements.

    nondet_dict: using isinstance to dispatch nondet type inside model
      functions does not work reliably. The type check fails even when
      the correct type is passed.

  So we are using the preprocessor, which works
      by expanding calls inline as user code:

    nondet_list: while loop with fresh nondet_*() per iteration.
      x = nondet_list(3, nondet_bool())  -->
        x: list[bool] = []; ...
        while i < size: x.append(nondet_bool()); i += 1

    nondet_dict: if-chain with concrete sequential keys to avoid O(N^2)
      solver explosion from symbolic key comparisons in contains/find_index.
      x = nondet_dict(3)  -->
        x: dict[int,int] = {}; ...
        if size >= 1: x[0] = nondet_int()
        if size >= 2: x[1] = nondet_int()
        if size >= 3: x[2] = nondet_int()

  Only direct assignments (x = nondet_list/dict(...)) are expanded.
  Once the frontend bugs are fixed, the preprocessor expansion
  can be removed and the fixes moved directly into this file.

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

# pylint: disable=undefined-variable
# `nondet_int`, `nondet_bool`, `__ESBMC_assume`, etc. are ESBMC
# intrinsics matched by name by the Python converter; they have no
# Python binding.

from typing import Any

# Shared default maximum size for nondet collections
_DEFAULT_NONDET_SIZE: int = 8


def _nondet_size(max_size: int) -> int:
    """
    Generate a non-deterministic size in range [0, max_size].

    Args:
        max_size: Maximum size (inclusive).

    Returns
    -------
    int
        A non-deterministic integer in [0, max_size].

    """
    size: int = nondet_int()
    __ESBMC_assume(size >= 0)
    __ESBMC_assume(size <= max_size)
    return size


def nondet_list(max_size: int = _DEFAULT_NONDET_SIZE, elem_type: Any = None) -> list:
    """
    Return a non-deterministic list with specified element type.

    Parameters
    ----------
    max_size
        Maximum size of the list (default: 8). The actual size will be in
        range [0, max_size].
    elem_type
        Value returned by type constructor for list elements
        (default: nondet_int()). Supported: nondet_int(), nondet_float(),
        nondet_bool(), nondet_str().

    Returns
    -------
    list
        A list with arbitrary size and contents of the specified type.

    Examples
    --------
        x = nondet_list()                                     # int list, size [0, 8]
        x = nondet_list(5)                                    # int list, size [0, 5]
        x = nondet_list(elem_type=nondet_float())             # float list, size [0, 8]
        x = nondet_list(max_size=10, elem_type=nondet_bool()) # bool list, size [0, 10]

    """
    # Default to nondet_int if no type specified
    if elem_type is None:
        elem_type = nondet_int()

    result: list = []
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        result.append(elem_type)
        i = i + 1

    return result


def nondet_dict(max_size: int = _DEFAULT_NONDET_SIZE,
                key_type: Any = None,
                value_type: Any = None) -> dict:
    """
    Return a non-deterministic dictionary with specified key and value types.

    The preprocessor expands this call inline with concrete sequential keys
    and fresh nondet values. This model function body is the fallback for
    non-expanded contexts (e.g. return values, nested exprs).

    Parameters
    ----------
    max_size
        Maximum size of the dictionary (default: 8). The actual size will be
        in range [0, max_size].
    key_type
        Value returned by type constructor for dictionary keys
        (default: nondet_int()). Supported: nondet_int(), nondet_str(),
        nondet_bool().
    value_type
        Value returned by type constructor for dictionary values
        (default: nondet_int()). Supported: nondet_int(), nondet_float(),
        nondet_bool(), nondet_str().

    Returns
    -------
    dict
        A dictionary with arbitrary size and contents of the specified types.

    Examples
    --------
        d = nondet_dict()                    # int->int dict, size [0, 8]
        d = nondet_dict(5)                   # int->int dict, size [0, 5]
        d = nondet_dict(key_type=nondet_str(), value_type=nondet_float())
        d = nondet_dict(max_size=10, key_type=nondet_int(), value_type=nondet_bool())

    """
    # Default to nondet_int if no types specified
    if key_type is None:
        key_type = nondet_int()
    if value_type is None:
        value_type = nondet_int()

    result: dict = {}
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        result[key_type] = value_type
        i = i + 1

    return result
