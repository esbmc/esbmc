# Operational model for the queue module.
# pylint: disable=unused-argument
# Single-threaded verification model: queue.Queue is backed by a plain
# Python list, which the frontend already supports (append / pop / pop(0) /
# __len__). The blocking semantics of put()/get() (the `block` and `timeout`
# arguments) are NOT modelled — there are no threads to block on under
# sequential symbolic execution. An unguarded get() on an empty queue pops
# from an empty list, which the list model reports as an "IndexError: pop from
# empty list" violation (rather than CPython's block-forever / queue.Empty);
# programs that guard with empty()/qsize() first verify cleanly. task_done() /
# join() are accepted no-ops. maxsize is tracked by full() but put() does not
# block on it.
from typing import Any


class Queue:
    """queue.Queue — list-backed FIFO (first-in, first-out)."""

    def __init__(self, maxsize: int = 0) -> None:
        self.items: list[Any]
        self.items = []
        self.maxsize: int = maxsize

    def put(self, item: Any, block: bool = True, timeout: Any = None) -> None:
        self.items.append(item)

    def put_nowait(self, item: Any) -> None:
        self.items.append(item)

    def get(self, block: bool = True, timeout: Any = None) -> Any:
        return self.items.pop(0)

    def get_nowait(self) -> Any:
        return self.items.pop(0)

    def qsize(self) -> int:
        return len(self.items)

    def empty(self) -> bool:
        return len(self.items) == 0

    def full(self) -> bool:
        return self.maxsize > 0 and len(self.items) >= self.maxsize

    def task_done(self) -> None:
        return None

    def join(self) -> None:
        return None


class LifoQueue:
    """queue.LifoQueue — list-backed LIFO (last-in, first-out / stack)."""

    def __init__(self, maxsize: int = 0) -> None:
        self.items: list[Any]
        self.items = []
        self.maxsize: int = maxsize

    def put(self, item: Any, block: bool = True, timeout: Any = None) -> None:
        self.items.append(item)

    def put_nowait(self, item: Any) -> None:
        self.items.append(item)

    def get(self, block: bool = True, timeout: Any = None) -> Any:
        return self.items.pop()

    def get_nowait(self) -> Any:
        return self.items.pop()

    def qsize(self) -> int:
        return len(self.items)

    def empty(self) -> bool:
        return len(self.items) == 0

    def full(self) -> bool:
        return self.maxsize > 0 and len(self.items) >= self.maxsize

    def task_done(self) -> None:
        return None

    def join(self) -> None:
        return None
