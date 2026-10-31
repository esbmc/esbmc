from typing import Optional


def make() -> dict[str, Optional[dict[str, str]]]:
    return {"k": {"t": "x"}}


extras = make()
h = extras["k"]
assert "t" in h
