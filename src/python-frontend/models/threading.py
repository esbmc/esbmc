"""Operational model for the standard ``threading`` module."""
# pylint: disable=unused-argument,unnecessary-pass,undefined-variable,function-redefined
# Stub for the ``threading`` module. ESBMC currently models the
# non-recursive mutex (``threading.Lock``) and ``threading.Thread``.
# Other names from the ``threading`` namespace are rejected at parse
# time by ``reject_unsupported_threading_usage`` in ``parser.py``.
#
# ``Lock.acquire`` / ``Lock.release`` are implemented in pure Python on
# top of ESBMC's atomic-section and assume intrinsics; the assume on
# ``self._locked == 0`` forces ESBMC's interleaving search to schedule
# the calling thread only on states where the lock is free, which is
# the same semantics ``pthread_mutex_lock_noassert`` uses in the C
# operational model.
#
# ``Thread`` is a thin skeleton: the constructor, ``start``, and ``join``
# are rewritten away by ``desugar_threading_thread`` in ``parser.py``
# before this code runs. The class still exists so ``Thread()`` parses
# and yields an instance with a ``_tid`` field the rewrite can store the
# spawned thread id into.


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


class Thread:
    """Skeleton matching ``threading.Thread``.

    The bodies here are never executed: ``parser.py`` rewrites every
    ``Thread(target=f, args=(...))`` construction, ``t.start()`` call, and
    ``t.join()`` call into spawn-thread / pthread-join intrinsics before
    these methods are invoked. The class only needs to exist so the
    construction parses to an instance with a ``_tid`` field the rewrite
    stores the spawned thread id into.
    """

    def __init__(self) -> None:
        self._tid: int = 0

    def start(self) -> None:
        """Replaced by ``desugar_threading_thread`` — never executes."""
        pass

    def join(self) -> None:
        """Replaced by ``desugar_threading_thread`` — never executes."""
        pass
