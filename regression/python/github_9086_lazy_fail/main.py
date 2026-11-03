# GitHub #9086: in an and/or tail the TypeError is raised only when the tail
# runs.
from typing import Optional


def f(c: bool) -> bool:
    s: Optional[str] = None
    try:
        return c and int(s) > 0
    except TypeError:
        assert not c
        return False


f(nondet_bool())
