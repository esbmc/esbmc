from typing import Optional


def make() -> dict[str, Optional[dict[str, str]]]:
    return {"k": None}


extras = make()
h = extras["k"]
if "t" in h:
    pass
