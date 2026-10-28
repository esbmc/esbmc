from typing import Optional


def lookup(extras: dict[str, Optional[dict[str, str]]]) -> None:
    h = extras["k"]
    if "t" in h:
        pass


e: dict[str, Optional[dict[str, str]]] = {}
e["k"] = None
lookup(e)
