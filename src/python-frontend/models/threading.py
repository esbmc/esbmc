"""Operational model for the standard ``threading`` module."""
# pylint: disable=unused-argument,unnecessary-pass,undefined-variable,function-redefined
# Stub for the ``threading`` module. ``Lock`` is fully modelled here;
# ``Thread`` is modelled at the AST-rewrite layer in ``parser.py``
# (``lower_threading_thread_usage``) plus three C intrinsics in
# ``src/c2goto/library/pthread_lib.c``. Any other name from the
# ``threading`` namespace is rejected at parse time by
# ``reject_unsupported_threading_usage`` in ``parser.py``.
#
# ``Lock.acquire`` / ``Lock.release`` are implemented in pure Python on
# top of ESBMC's atomic-section and assume intrinsics; the assume on
# ``self._locked == 0`` forces ESBMC's interleaving search to schedule
# the calling thread only on states where the lock is free, which is
# the same semantics ``pthread_mutex_lock_noassert`` uses in the C
# operational model.
#
# ``Thread`` here is a thin skeleton with no methods: every
# ``Thread(target=...)`` construction, ``.start()`` call, and
# ``.join()`` call is rewritten by ``lower_threading_thread_usage``
# before this class is ever instantiated. The skeleton's only purpose
# is to satisfy the Python frontend's class-instantiation machinery
# when the rewritten construction lowers to ``Thread()`` (no args).


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

    Constructor takes no parameters: the AST pre-pass strips
    ``target=``/``args=`` from every Thread construction before the
    Python frontend reaches this class, so the skeleton only has to
    exist as a placeholder. ``start()``/``join()`` are likewise
    rewritten by ``lower_threading_thread_usage`` and never invoked
    on this skeleton in well-formed programs.

    The method bodies assert ``False`` to make any path that *does*
    reach them — e.g. a dynamic-lookup escape such as
    ``Cls = threading.Thread; Cls()`` that the validator did not see
    — fail loudly with a verification assertion rather than silently
    skipping the spawn.
    """

    def __init__(self) -> None:
        self._tid: int = 0

    def start(self) -> None:
        """Reached only when the AST rewrite failed to lower a Thread."""
        assert False, ("threading.Thread.start() reached without AST lowering "
                       "(dynamic Thread lookup?)")

    def join(self) -> None:
        """Reached only when the AST rewrite failed to lower a Thread."""
        assert False, ("threading.Thread.join() reached without AST lowering "
                       "(dynamic Thread lookup?)")
