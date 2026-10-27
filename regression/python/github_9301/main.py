from typing import Optional


def make() -> dict[str, Optional[dict[str, str]]]:
    extras: dict[str, Optional[dict[str, str]]] = {}
    extras["k"] = {"t": "x"}
    return extras


extras = make()
h = extras["k"]
assert "t" in h
