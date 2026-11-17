# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2026 Raphaël Monat, Inria
# SPDX-FileCopyrightText: 2026-... The SV-Benchmarks Community
#
# SPDX-License-Identifier: Apache-2.0

import random
import string
import collections.abc
import types
from typing import Any, Callable, Literal, Union, get_args, get_origin

def nondet_bool():
    return bool(random.randint(0, 1))

def nondet_int() -> int:
    # unbounded int
    i = 0
    while nondet_bool():
        i += 1
    if nondet_bool():
        i = -i
    return i

def nondet_list(nondet_elem):
    l = []
    while nondet_bool():
        l.append(nondet_elem())
    return l

def nondet_dict(nondet_key, nondet_value):
    d = dict()
    while nondet_bool():
        d[nondet_key()] = nondet_value()
    return d

def nondet_str():
    # Source - https://stackoverflow.com/a/23728630
    # Posted by Randy Marsh, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-01, License - CC BY-SA 4.0
    # Note that this is the concrete implementation, but it could be changed to include *any* kind of caracter
    return ''.join(random.choice(string.printable) for _ in range(abs(nondet_int())))

def nondet_float():
    import sys
    if nondet_bool():
        return random.choice([float('inf'), float('-inf'), float('nan'), float('-0.')])
    return random.uniform(-sys.float_info.max, sys.float_info.max)

def _matches_type(value, hint) -> bool:
    if hint is Any:
        return True
    if hint is None or hint is type(None):
        return value is None
    origin = get_origin(hint)
    if origin is None:
        # Plain class, e.g. int or list
        return isinstance(value, hint)
    args = get_args(hint)
    if origin is Union or origin is types.UnionType:
        return any(_matches_type(value, a) for a in args)
    if origin is Literal:
        return any(value == a and type(value) is type(a) for a in args)
    if not isinstance(value, origin):
        return False
    if not args:
        return True
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches_type(v, args[0]) for v in value)
        if args == ((),):
            return len(value) == 0
        return len(value) == len(args) and all(_matches_type(v, a) for v, a in zip(value, args))
    if origin is collections.abc.Callable:
        # Signatures are not checked, only callability
        return True
    if issubclass(origin, collections.abc.Mapping):
        key_hint, value_hint = args
        return all(_matches_type(k, key_hint) and _matches_type(v, value_hint) for k, v in value.items())
    if issubclass(origin, collections.abc.Iterable):
        (elem_hint,) = args
        return all(_matches_type(v, elem_hint) for v in value)
    raise NotImplementedError(f"check_type: unsupported type hint {hint!r}")

def check_type(value, hint):
    """Raise a TypeError if value is not of the type described by hint (e.g. list[list[int]])."""
    if not _matches_type(value, hint):
        raise TypeError(f"expected value of type {hint!r}, got {value!r}")
