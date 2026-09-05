"""
Operational model for ESBMC's non-deterministic Python collection generators.

Every builder here is *monomorphic*: it constructs one element type only, and
calls the scalar generator (``nondet_int()`` and friends) afresh for each
element, so distinct indices hold independent symbolic values.

Monomorphism is a requirement, not a style choice.  A single builder that
dispatches on a type tag --

    if kind == 0: result.append(nondet_int())
    else:         result.append(nondet_float())

-- is silently mistyped by the frontend: every element degrades to the first
branch's type, so a float list cannot hold 0.5 and properties over it are
proved vacuously.  One function per element type avoids the dispatch entirely.

The preprocessor rewrites ``nondet_list``/``nondet_dict`` to the matching
builder below (``_parse_nondet_collection_call`` in
``preprocessor/loop_mixin.py``), so a call in *any* expression position -- an
annotated assignment, a return, a call argument -- gets fresh elements.  A
module that defines its own ``nondet_list``/``nondet_dict`` is left alone; an
*imported* one is still intercepted, which is what SV-COMP harnesses expect.
The public ``nondet_list``/``nondet_dict`` at the end of this file exist so the
name resolves for ``from esbmc import nondet_list``; direct calls are rewritten
before they reach those bodies.

Dictionary keys are concrete and sequential rather than symbolic: symbolic keys
turn every ``contains``/``find_index`` check into a linear scan of symbolic
comparisons, which is quadratic in the solver, while concrete keys make each
check trivially decidable and leave the values fully non-deterministic.  An
int-keyed dict is inserted in a loop and so honours any requested bound; a
str-keyed one is capped at ``_MAX_NONDET_STR_KEYS`` by the key table, and a
bool-keyed one at two, since that is how many bool keys exist.  Both clamp the
assumed size to their cap -- otherwise a size above it would be assumed
reachable but never populated.

See esbmc/esbmc#7575 for the element-reuse defect this file's shape addresses.

USAGE:
    # Lists:
    x = nondet_list()                                     # int list, size [0, 8]
    x = nondet_list(5)                                    # int list, size [0, 5]
    x = nondet_list(elem_type=nondet_float())             # float list, size [0, 8]
    x = nondet_list(max_size=10, elem_type=nondet_bool()) # bool list, size [0, 10]

    # Dictionaries:
    d = nondet_dict()                                     # int->int dict, size [0, 8]
    d = nondet_dict(5)                                    # int->int dict, size [0, 5]
    d = nondet_dict(key_type=nondet_str(), value_type=nondet_float())
    d = nondet_dict(max_size=10, key_type=nondet_int(), value_type=nondet_bool())

    An int-keyed dict honours any `max_size`; a str-keyed one is capped at 8
    entries and a bool-keyed one at 2 (see `_MAX_NONDET_STR_KEYS`).
"""

# pylint: disable=undefined-variable,unused-argument
# `nondet_int`, `nondet_bool`, `__ESBMC_assume`, etc. are ESBMC
# intrinsics matched by name by the Python converter; they have no
# Python binding.

from typing import Any

# Shared default maximum size for nondet collections. Mirrored by
# `_DEFAULT_NONDET_COLLECTION_SIZE` in preprocessor/loop_mixin.py, which applies
# it when a call omits the bound.
_DEFAULT_NONDET_SIZE: int = 8

# Length of the str key table below, and so the largest str-keyed dict this
# model can build. int keys have no such limit: they are inserted in a loop.
_MAX_NONDET_STR_KEYS: int = 8

# bool admits exactly two keys, however large the requested bound.
_MAX_NONDET_BOOL_KEYS: int = 2


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


def _nondet_dict_size(max_size: int) -> int:
    """
    Generate a non-deterministic dict size in range [0, max_size].

    Args:
        max_size: Maximum entry count (inclusive).

    Returns
    -------
    int
        A non-deterministic integer in [0, max_size].

    """
    # Generated here rather than through `_nondet_size` so the symbol stays in
    # dict scope: test-case generation reads the owning function to tell a dict
    # entry count from a list length.
    size: int = nondet_int()
    __ESBMC_assume(size >= 0)
    __ESBMC_assume(size <= max_size)
    return size

def _nondet_str_key_bound(max_size: int) -> int:
    """Clamp a requested size to the number of str keys this model can build."""
    # Clamped by hand rather than with `min`: routing two ints through the
    # builtin's operational model costs ~17s of conversion time on every
    # Python program, because this file is converted whether or not it is used.
    bound: int = max_size
    if bound > _MAX_NONDET_STR_KEYS:  # pylint: disable=consider-using-min-builtin
        bound = _MAX_NONDET_STR_KEYS
    return bound


def _nondet_bool_key_bound(max_size: int) -> int:
    """Clamp a requested size to the two keys a bool admits."""
    bound: int = max_size
    if bound > _MAX_NONDET_BOOL_KEYS:  # pylint: disable=consider-using-min-builtin
        bound = _MAX_NONDET_BOOL_KEYS
    return bound



def _nondet_list_int(max_size: int) -> list:
    """Return a list of size [0, max_size] whose elements are independent ints."""
    result: list[int] = []
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        result.append(nondet_int())
        i = i + 1

    return result


def _nondet_list_float(max_size: int) -> list:
    """Return a list of size [0, max_size] whose elements are independent floats."""
    result: list[float] = []
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        result.append(nondet_float())
        i = i + 1

    return result


def _nondet_list_bool(max_size: int) -> list:
    """Return a list of size [0, max_size] whose elements are independent bools."""
    result: list[bool] = []
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        result.append(nondet_bool())
        i = i + 1

    return result


def _nondet_list_str(max_size: int) -> list:
    """Return a list of size [0, max_size] whose elements are independent strs."""
    result: list[str] = []
    size: int = _nondet_size(max_size)

    i: int = 0
    while i < size:
        result.append(nondet_str())
        i = i + 1

    return result


def _nondet_dict_int_int(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping int keys to independent int values."""
    result: dict[int, int] = {}
    size: int = _nondet_dict_size(max_size)

    i: int = 0
    while i < size:
        result[i] = nondet_int()
        i = i + 1

    return result


def _nondet_dict_int_float(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping int keys to independent float values."""
    result: dict[int, float] = {}
    size: int = _nondet_dict_size(max_size)

    i: int = 0
    while i < size:
        result[i] = nondet_float()
        i = i + 1

    return result


def _nondet_dict_int_bool(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping int keys to independent bool values."""
    result: dict[int, bool] = {}
    size: int = _nondet_dict_size(max_size)

    i: int = 0
    while i < size:
        result[i] = nondet_bool()
        i = i + 1

    return result


def _nondet_dict_int_str(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping int keys to independent str values."""
    result: dict[int, str] = {}
    size: int = _nondet_dict_size(max_size)

    i: int = 0
    while i < size:
        result[i] = nondet_str()
        i = i + 1

    return result


def _nondet_dict_str_int(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping str keys to independent int values."""
    # A local table, not a module constant: a module-scope list literal is
    # built by every Python program ESBMC runs, whether or not it uses a dict.
    # `str(i)` cannot replace it -- inside the loop it yields the same key for
    # every index, collapsing the dict to one entry.
    keys: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"]
    result: dict[str, int] = {}
    size: int = _nondet_dict_size(_nondet_str_key_bound(max_size))

    i: int = 0
    while i < size:
        result[keys[i]] = nondet_int()
        i = i + 1

    return result


def _nondet_dict_str_float(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping str keys to independent float values."""
    # A local table, not a module constant: a module-scope list literal is
    # built by every Python program ESBMC runs, whether or not it uses a dict.
    # `str(i)` cannot replace it -- inside the loop it yields the same key for
    # every index, collapsing the dict to one entry.
    keys: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"]
    result: dict[str, float] = {}
    size: int = _nondet_dict_size(_nondet_str_key_bound(max_size))

    i: int = 0
    while i < size:
        result[keys[i]] = nondet_float()
        i = i + 1

    return result


def _nondet_dict_str_bool(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping str keys to independent bool values."""
    # A local table, not a module constant: a module-scope list literal is
    # built by every Python program ESBMC runs, whether or not it uses a dict.
    # `str(i)` cannot replace it -- inside the loop it yields the same key for
    # every index, collapsing the dict to one entry.
    keys: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"]
    result: dict[str, bool] = {}
    size: int = _nondet_dict_size(_nondet_str_key_bound(max_size))

    i: int = 0
    while i < size:
        result[keys[i]] = nondet_bool()
        i = i + 1

    return result


def _nondet_dict_str_str(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping str keys to independent str values."""
    # A local table, not a module constant: a module-scope list literal is
    # built by every Python program ESBMC runs, whether or not it uses a dict.
    # `str(i)` cannot replace it -- inside the loop it yields the same key for
    # every index, collapsing the dict to one entry.
    keys: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7"]
    result: dict[str, str] = {}
    size: int = _nondet_dict_size(_nondet_str_key_bound(max_size))

    i: int = 0
    while i < size:
        result[keys[i]] = nondet_str()
        i = i + 1

    return result


def _nondet_dict_bool_int(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping bool keys to independent int values."""
    result: dict[bool, int] = {}
    size: int = _nondet_dict_size(_nondet_bool_key_bound(max_size))

    i: int = 0
    while i < size:
        result[i % 2 == 0] = nondet_int()
        i = i + 1

    return result


def _nondet_dict_bool_float(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping bool keys to independent float values."""
    result: dict[bool, float] = {}
    size: int = _nondet_dict_size(_nondet_bool_key_bound(max_size))

    i: int = 0
    while i < size:
        result[i % 2 == 0] = nondet_float()
        i = i + 1

    return result


def _nondet_dict_bool_bool(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping bool keys to independent bool values."""
    result: dict[bool, bool] = {}
    size: int = _nondet_dict_size(_nondet_bool_key_bound(max_size))

    i: int = 0
    while i < size:
        result[i % 2 == 0] = nondet_bool()
        i = i + 1

    return result


def _nondet_dict_bool_str(max_size: int) -> dict:
    """Return a dict of size [0, max_size] mapping bool keys to independent str values."""
    result: dict[bool, str] = {}
    size: int = _nondet_dict_size(_nondet_bool_key_bound(max_size))

    i: int = 0
    while i < size:
        result[i % 2 == 0] = nondet_str()
        i = i + 1

    return result


def nondet_list(max_size: int = _DEFAULT_NONDET_SIZE, elem_type: Any = None) -> list:
    """
    Return a non-deterministic list.

    Kept so the name resolves -- ``from esbmc import nondet_list`` binds to this
    definition rather than to esbmc.py's empty stub. Direct calls never reach
    the body: the preprocessor rewrites them to the builder for their element
    type. Elements are independent ints.
    """
    return _nondet_list_int(max_size)


def nondet_dict(max_size: int = _DEFAULT_NONDET_SIZE,
                key_type: Any = None,
                value_type: Any = None) -> dict:
    """
    Return a non-deterministic dictionary.

    Kept so the name resolves (see `nondet_list`); direct calls never reach the
    body. Keys and values are independent ints.
    """
    return _nondet_dict_int_int(max_size)
