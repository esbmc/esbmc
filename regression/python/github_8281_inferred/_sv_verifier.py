# Trimmed from sv-benchmarks' _sv_verifier.py (!1792): check_type and the
# reflection-based helper it calls. The frontend must not convert these
# bodies; check_type's meaning is supplied at the call site.
from typing import Any, Literal, Union, get_args, get_origin
import collections.abc
import types


def _matches_type(value, hint) -> bool:
    if hint is Any:
        return True
    if hint is None or hint is type(None):
        return value is None
    origin = get_origin(hint)
    if origin is None:
        return isinstance(value, hint)
    args = get_args(hint)
    if origin is Union or origin is types.UnionType:
        return any(_matches_type(value, a) for a in args)
    if origin is Literal:
        return any(value == a and type(value) is type(a) for a in args)
    if issubclass(origin, collections.abc.Mapping):
        key_hint, value_hint = args
        return all(
            _matches_type(k, key_hint) and _matches_type(v, value_hint)
            for k, v in value.items()
        )
    raise NotImplementedError("unsupported type hint")


def check_type(value, hint):
    """Raise a TypeError if value is not of the type described by hint."""
    if not _matches_type(value, hint):
        raise TypeError("expected value of type")
