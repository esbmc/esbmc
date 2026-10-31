from typing import Optional


def make() -> dict[str, Optional[dict[str, str]]]:
    d: dict[str, Optional[dict[str, str]]] = {"k": {"t": "x"}, "n": None}
    return d


extras = make()
h = extras["k"]
assert "t" in h
assert extras["n"] is None
