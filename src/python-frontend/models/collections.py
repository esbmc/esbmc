# Operational model for collections module
# pylint: disable=function-redefined  # intentional stdlib shadow for ESBMC models
# pylint: disable=keyword-arg-before-vararg,unused-argument
# defaultdict's signature is pinned: callers in regression/python/
# (e.g. github_3841_*) pass `default_factory` positionally as in
# `defaultdict(int)`, so it must come first; and the no-arg form
# `defaultdict()` is recognised by preprocessor.py:_get_defaultdict_factory,
# so the default value cannot be removed. *args / **kwargs are part of
# the API contract; the body silently ignores them by design (see the
# docstring below for the verification approximation).

from typing import Any, Optional


def defaultdict(default_factory: Optional[Any] = None, *args, **kwargs) -> dict:
    """
    Create a defaultdict - modeled as a plain dict for verification purposes.

    Approximations:
    - The default_factory is tracked by the preprocessor and used to insert
      missing-key defaults; it is not stored on the dict object itself.
    - Initial data passed as a positional mapping or iterable (*args) and any
      keyword arguments (**kwargs) are accepted but silently ignored. Pre-populated
      defaultdicts are not modeled; only keys written explicitly in the program
      will be present in the verification model.
    """
    return {}


def deque(iterable: list[int] = None, maxlen: int = -1) -> list[int]:
    """collections.deque(iterable=()) — modelled as a plain list.

    Approximation: deque is treated as a list backing store, exposing the
    same `append` / `pop` / `__len__` / subscript surface that the
    frontend already supports for lists. `appendleft` / `popleft` /
    `extendleft` and the `maxlen` rollover are NOT modelled; programs
    relying on the FIFO end will need a richer model.
    """
    if iterable is None:
        return []
    result: list[int] = []
    i: int = 0
    n: int = len(iterable)
    while i < n:
        result.append(iterable[i])
        i = i + 1
    return result


def OrderedDict(*args, **kwargs) -> dict:
    """collections.OrderedDict — modelled as a plain dict.

    Insertion order is already preserved by the dict model, so the
    OrderedDict-specific methods (`move_to_end`) are the only divergence;
    `popitem` is already available via the dict interface.
    """
    return {}


class Counter:
    """Simplified Counter model: maps (int, int) keys to integer counts."""

    def __init__(self) -> None:
        """Initialize an empty counter with no recorded keys."""
        self.data: dict[tuple[int, int], int]
        self.data = {}
        self.count: int = 0

    def __getitem__(self, key: tuple[int, int]) -> int:
        """Return the count for *key*, or 0 if *key* has not been recorded."""
        try:
            return self.data[key]
        except KeyError:
            return 0

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """Set the count for *key*, tracking unique-key arrivals via ``count``."""
        try:
            self.data[key]
        except KeyError:
            self.count = self.count + 1
        self.data[key] = value

    def values(self) -> list[int]:
        """Return all recorded counts."""
        return self.data.values()

    def __bool__(self) -> bool:
        """Return True iff at least one key has been recorded."""
        return self.count != 0
