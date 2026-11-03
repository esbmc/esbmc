from typing import Optional


class M:
    def __init__(self) -> None:
        extras: dict[str, Optional[dict[str, str]]] = {}
        extras["k"] = None
        self.extras: dict[str, Optional[dict[str, str]]] = extras

    def lookup(self) -> None:
        h = self.extras["k"]
        if "t" in h:
            pass


M().lookup()
