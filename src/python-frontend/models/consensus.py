# pylint: disable=redefined-builtin,unused-argument
# `hash` here intentionally shadows the Python built-in: it is the
# operational model ESBMC uses to verify Python programs, so it must
# match the built-in name exactly. Argument names on the abstract stub
# are part of the API contract matched by ESBMC's Python converter,
# even when the body does not reference them.

# Stubs used for consensus specification verification


def hash(data: bytes) -> bytes:
    # Real SHA-256 over a symbolic `data` is intractable for the solver
    # (64 rounds of bit rotation/addition per call). This gives the real
    # Bytes32 shape instead: 32 independent nondet bytes, with no claim of
    # determinism or any relation to `data`.
    return nondet_bytes(32)
