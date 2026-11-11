# A dict value stored from a loop local keeps its pointer targets after
# the frame ends.
from typing import Optional


class M:

    def __init__(self) -> None:
        extras: dict[str, Optional[dict[str, str]]] = {}
        i: int = 0
        while i < 2:
            inner: Optional[dict[str, str]] = None
            if nondet_bool():
                d: dict[str, str] = {}
                inner = d
            if i == 0:
                extras["k"] = inner
            i += 1
        self.extras: dict[str, Optional[dict[str, str]]] = extras

    def lookup(self) -> None:
        h = self.extras["k"]
        if h is not None:
            assert "t" not in h


M().lookup()
