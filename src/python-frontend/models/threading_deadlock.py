"""Deadlock-aware operational model for the standard ``threading`` module.

Loaded in place of ``threading.py`` when ESBMC is invoked with
``--deadlock-check``. The Python frontend's ``parser.py`` performs the
swap, mirroring the C preprocessor swap of ``pthread_mutex_lock`` to
``pthread_mutex_lock_check`` in ``src/clang-c-frontend/c_preprocess.cpp``.

The acquire path mirrors ``pthread_mutex_lock_check`` in
``src/c2goto/library/pthread_lib.c``: read the lock field inside an
atomic block, take the lock on the unlocked branch, or bump
``__ESBMC_blocked_threads_count`` and assert the global deadlock
predicate on the locked branch. The ``__ESBMC_assume(unlocked)`` after
the atomic-end kills the locked branch on this thread, but symex's
interleaving search observes the bumped counter on alternative
schedules where every running thread reaches the locked branch — that
is where the predicate fires.

The release path mirrors ``pthread_mutex_unlock_check``: assert the
lock was held before clearing the field, so a release-without-acquire
bug is caught under the same flag whose job it is to find concurrency
errors.
"""
# pylint: disable=unused-argument,unnecessary-pass,undefined-variable,function-redefined


class Lock:
    """Non-recursive mutex; mirrors ``threading.Lock`` with deadlock detection."""

    def __init__(self) -> None:
        self._locked: int = 0

    def acquire(self) -> None:
        """Block until the lock is free, then mark it held.

        On the blocked branch, register the thread as blocked and assert
        that not every running thread is now blocked. Symex's
        interleaving search lifts the bumped counter into schedules
        where the deadlock predicate becomes true and the assertion
        fires; on this thread's own path, ``__ESBMC_assume(unlocked)``
        kills the blocked branch after the predicate has been checked.
        """
        __ESBMC_atomic_begin()
        unlocked: bool = (self._locked == 0)
        if unlocked:
            self._locked = 1
        else:
            __ESBMC_pylock_block_and_check()
        __ESBMC_atomic_end()
        __ESBMC_assume(unlocked)

    def release(self) -> None:
        """Mark the lock free; assert it was held.

        The assert is consumed by ESBMC's parser.py via ast.parse() and
        lowered to a verification claim, so `python -O` byte-code
        stripping never applies — the nosec marker silences Bandit's
        generic B101 finding, which does not model that pipeline.
        """
        __ESBMC_atomic_begin()
        assert self._locked == 1, "must hold lock upon unlock"  # nosec B101
        self._locked = 0
        __ESBMC_atomic_end()


class Thread:
    """Skeleton matching ``threading.Thread``; identical to the assume-only variant.

    Every ``Thread(target=...)`` construction, ``.start()`` call, and
    ``.join()`` call is rewritten by ``lower_threading_thread_usage``
    in ``parser.py`` before this class is ever instantiated. The
    skeleton's only purpose is to satisfy the Python frontend's
    class-instantiation machinery when the rewritten construction
    lowers to ``Thread()`` (no args).
    """

    def __init__(self) -> None:
        self._tid: int = 0

    def start(self) -> None:
        """Reached only when the AST rewrite failed to lower a Thread."""
        __ESBMC_assert(
            False,
            "threading.Thread.start() reached without AST lowering "
            "(dynamic Thread lookup?)",
        )

    def join(self) -> None:
        """Reached only when the AST rewrite failed to lower a Thread."""
        __ESBMC_assert(
            False,
            "threading.Thread.join() reached without AST lowering "
            "(dynamic Thread lookup?)",
        )
