# A dict value read after its source local's frame has ended keeps its
# pointer targets (#9169 regressed this).
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
            assert "t" not in h


M().lookup()
