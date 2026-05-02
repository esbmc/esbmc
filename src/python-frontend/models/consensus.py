# pylint: disable=redefined-builtin
# `hash` here intentionally shadows the Python built-in: it is the
# operational model ESBMC uses to verify Python programs, so it must
# match the built-in name exactly.

# Stubs used for consensus specification verification


def hash(data: bytes) -> int:
    return 42
