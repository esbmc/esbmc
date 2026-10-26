# The read keeps the empty dict, so membership of "t" fails.
from typing import Optional


class M:

    def __init__(self) -> None:
        extras: dict[str, Optional[dict[str, str]]] = {}
        inner: Optional[dict[str, str]] = None
        if nondet_bool():
            d: dict[str, str] = {}
            inner = d
        extras["k"] = inner
        self.extras: dict[str, Optional[dict[str, str]]] = extras

    def lookup(self) -> None:
        h = self.extras["k"]
        if h is not None:
            assert "t" in h


M().lookup()
