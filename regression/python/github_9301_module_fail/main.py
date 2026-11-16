from typing import Optional

extras: dict[str, Optional[dict[str, str]]] = {}
extras["k"] = None
h = extras["k"]
if "t" in h:
    pass
