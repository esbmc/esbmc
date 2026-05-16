"""Operational model for the standard ``threading`` module."""
# pylint: disable=unused-argument,unnecessary-pass,undefined-variable,function-redefined
# Stub for the ``threading`` module. ESBMC currently models only the
# non-recursive mutex (``threading.Lock``). Any other name from the
# ``threading`` namespace is rejected at parse time by
# ``reject_unsupported_threading_usage`` in ``parser.py``.
#
# ``Lock.acquire`` / ``Lock.release`` are implemented in pure Python on
# top of ESBMC's atomic-section and assume intrinsics; the assume on
# ``self._locked == 0`` forces ESBMC's interleaving search to schedule
# the calling thread only on states where the lock is free, which is
# the same semantics ``pthread_mutex_lock_noassert`` uses in the C
# operational model.


class Lock:
    """Non-recursive mutex; mirrors ``threading.Lock``."""

    def __init__(self) -> None:
        self._locked: int = 0

    def acquire(self) -> None:
        """Block until the lock is free, then mark it held."""
        __ESBMC_atomic_begin()
        __ESBMC_assume(self._locked == 0)
        self._locked = 1
        __ESBMC_atomic_end()

    def release(self) -> None:
        """Mark the lock free."""
        __ESBMC_atomic_begin()
        self._locked = 0
        __ESBMC_atomic_end()
