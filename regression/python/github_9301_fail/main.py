from typing import Optional


def make() -> dict[str, Optional[dict[str, str]]]:
    extras: dict[str, Optional[dict[str, str]]] = {}
    extras["k"] = None
    return extras


extras = make()
h = extras["k"]
if "t" in h:
    pass
