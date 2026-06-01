# pylint: disable=redefined-builtin,unused-argument
# `hash` here intentionally shadows the Python built-in: it is the
# operational model ESBMC uses to verify Python programs, so it must
# match the built-in name exactly. Argument names on the abstract stub
# are part of the API contract matched by ESBMC's Python converter,
# even when the body does not reference them.

# Stubs used for consensus specification verification


def hash(data: bytes) -> int:
    return 42
